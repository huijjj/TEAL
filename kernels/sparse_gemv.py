# adapted from deja vu

from typing import Optional

import torch
import triton
import triton.language as tl
def init_to_zero(*names):
    def init_func(nargs):
        for name in names:
            nargs[name].zero_()
    return init_func

# NOTE: will need to warm up kernels each time, triton autotune caching isn't a thing right now

configs=[
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")), 

    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),

    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),


    # Llama 3 variants can use BLOCK_N >= 1024
    # triton.Config({"BLOCK_M": 128, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    # triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    # triton.Config({"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    # triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    # triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
]

@triton.autotune(
    configs=configs,
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE", "SPARSITY_BIN"],
)
@triton.jit
def splitk_sparse_gemv_kernel(
    Y, # Pointers to matrices
    A, X, threshold,
    # Matrix dimensions
    N, M,
    CACHE_KEY_N, CACHE_KEY_M,
    # Meta-parameters
    BATCHSIZE: tl.constexpr, SPARSITY_BIN: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    start_n = tl.program_id(0)
    start_m = tl.program_id(1)
    # now compute the block that each program will go through
    # rn (resp. rm) denotes a range of indices for rows (resp. col) of A
    
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    A_ptr = A + (rm[:, None] * N + rn[None, :])
    X_ptr = X + rm
    Y_ptr = Y + rn
    
    # eviction policy go brrr
    if BATCHSIZE == 1:
        x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy='evict_last') # reuse x across threadblocks
        idx = tl.abs(x0) > threshold
        # selectively load weight rows
        a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy='evict_first') # only load weights once per threadblock
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.atomic_add(Y_ptr, acc0, mask=rn < N)


# NOTE: assumes that weight is column major
def splitk_sparse_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
    sparsity_bin: int
) -> torch.Tensor:
    """
    Compute y = sparse(X) @ weight.
    :param x: input tensor [1, 1, Z]
    :param weight: weight matrix [N, Z]
    :param threshold: threshold for the absolute value of x
    :param sparsity_bin: sparsity level to get tuned kernel
    :return: result tensor y
    """
    N, Z = weight.shape
    beam_width, seq_len, _ = x.shape
    assert x.shape[2] == Z
    x = x.contiguous()
    
    assert weight.stride(1) > 1, "weight should be column major"

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(Z, META["BLOCK_M"]),
    )  # noqa

    output = torch.empty(
        beam_width,
        seq_len,
        N,
        device=x.device,
        dtype=torch.float16,
    )


    kernel = splitk_sparse_gemv_kernel
    kernel[grid](
        output,  # data ptrs
        weight,
        x,
        threshold,
        N,  # shapes
        Z,
        N // 16,  # key for triton cache (limit number of compilations)
        Z // 16,
        beam_width,  # BATCHSIZE
        sparsity_bin, # SPARSITY_BIN
        # can't use kwargs because auto-tuner requires args
    )

    if x.dtype is not output.dtype:
        print(f"Warning: incuring dtype conversion overhead since input dtype is not torch.float16. Detected dtype: {x.dtype}. ")
        return output.to(dtype=x.dtype)

    return output


@triton.autotune(configs=[
    triton.Config({"BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
], key=["M", "N", "K"])
@triton.jit
def splitk_sparse_gemm_kernel(
    x_ptr, # [M, K]
    w_ptr, # [N, K]
    o_ptr, # [M, N]
    threshold,
    M, N, K,
    x_M_stride, x_K_stride,
    w_N_stride, w_K_stride,
    o_M_stride, o_N_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    PID = tl.program_id(0)
    num_blocks_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    
    block_start_M = (PID//num_blocks_along_N)*BLOCK_SIZE_M
    M_offset = (block_start_M + tl.arange(0, BLOCK_SIZE_M))
    M_mask = M_offset < M

    block_start_N = (PID%num_blocks_along_N)*BLOCK_SIZE_N
    N_offset = (block_start_N + tl.arange(0, BLOCK_SIZE_N))
    N_mask = N_offset < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        K_offset = (k*BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        K_mask = K_offset < K

        x_block = tl.load(x_ptr + M_offset[:, None]*x_M_stride + K_offset[None,:]*x_K_stride, mask=(M_mask[:, None] & K_mask[None,:]), other=0.0) # [BLOCK_SIZE_M, BLOCK_SIZE_K]
        idx = tl.abs(x_block.sum(axis=0)) > threshold # [BLOCK_SIZE_K]

        w_block = tl.load(w_ptr + N_offset[:, None]*w_N_stride + K_offset[None, :]*w_K_stride, mask=(N_mask[:, None] & (K_mask & idx)[None, :]), other=0.0)
        
        acc = tl.dot(x_block, w_block.T, acc=acc)

    tl.store(o_ptr + M_offset[:, None]*o_M_stride + N_offset[None, :]*o_N_stride, acc.to(tl.bfloat16), mask=(M_mask[:, None] & N_mask[None, :]))


def splitk_sparse_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
    sparsity_bin: int
):
    # Compute y = sparse(X) @ weight.
    # :param x: input tensor [BS, 1, Z]
    # :param weight: weight matrix [N, Z]
    # :param threshold: threshold for the absolute value of x
    # :param sparsity_bin: sparsity level to get tuned kernel
    # :return: result tensor y

    N, K = weight.shape
    BS, SL, _ = x.shape
    assert x.shape[2] == K
    assert SL == 1
    x = x.view(-1, K)
    
    # assert weight.stride(1) > 1, "weight should be column major"

    output = torch.empty(BS*SL, N, device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(BS*SL, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
    splitk_sparse_gemm_kernel[grid](
        x,
        weight,
        output,
        threshold,
        BS*SL, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BS
    )

    return output.reshape(BS, SL, -1)

# fused implementation of qkv with three thresholds
# is unnecessary for uniform but is needed for block-wise greedy
@triton.autotune(
    configs=configs, 
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE", "SPARSITY_BIN"],
)
@triton.jit
def qkv_kernel(
    Y,  # Pointers to output matrices
    A, 
    X, 
    threshold_q, threshold_k, threshold_v,
    # Matrix dimensions
    N, N_q, N_kv, M,
    CACHE_KEY_N, CACHE_KEY_M,
    # Meta-parameters
    BATCHSIZE: tl.constexpr, SPARSITY_BIN: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    start_n = tl.program_id(0)
    start_m = tl.program_id(1)

    is_q = start_n * BLOCK_N < N_q
    is_v = N_q + N_kv <= start_n * BLOCK_N

    
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    rn = start_n*BLOCK_N + tl.arange(0, BLOCK_N)
        
    A_ptr = A + rm[:, None] * N + rn[None, :]

    X_ptr = X + rm

    Y_ptr = Y + rn

    threshold = tl.where(is_q, threshold_q, tl.where(is_v, threshold_v, threshold_k))
    
    if BATCHSIZE == 1:
        x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy='evict_last')
        idx = tl.abs(x0) > threshold
        a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy='evict_first')
        acc = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
    
    # rematerialize to reduce register pressure

    rn = start_n*BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < N

    tl.atomic_add(Y_ptr, acc, mask=mask_n)

def qkv_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold_q: float,
    threshold_k: float,
    threshold_v: float,
    sparsity_bin: int,
    kv_size: int
):
    N, Z = weight.shape
    beam_width, seq_len, _ = x.shape
    assert x.shape[2] == Z
    x = x.contiguous()
    
    assert weight.stride(1) > 1, "weights should be column major"

    N_q = N - 2*kv_size
    N_k = kv_size

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(Z, META["BLOCK_M"]),
    )

    output = torch.empty(beam_width, seq_len, N, device=x.device, dtype=torch.float16)

    qkv_kernel[grid](
        output,
        weight,
        x,
        threshold_q, threshold_k, threshold_v,
        N, N_q, N_k, Z,
        N // 16, Z // 16,
        beam_width,
        sparsity_bin,
    )

    if x.dtype is not output.dtype:
        print(f"Warning: incurring dtype conversion overhead. Input dtype: {x.dtype}")
        return output.to(dtype=x.dtype)

    return output


@triton.autotune(configs=[
    triton.Config({"BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
], key=["M", "N", "K"])
@triton.jit
def qkv_gemm_kernel(
    x_ptr, # [M, K]
    w_ptr, # [N, K]
    o_ptr, # [M, N]
    threshold_q,
    threshold_k,
    threshold_v,
    M, N, K,
    N_q, N_kv,
    x_M_stride, x_K_stride,
    w_N_stride, w_K_stride,
    o_M_stride, o_N_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    PID = tl.program_id(0)
    num_blocks_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    
    block_start_M = (PID//num_blocks_along_N)*BLOCK_SIZE_M
    M_offset = (block_start_M + tl.arange(0, BLOCK_SIZE_M))
    M_mask = M_offset < M

    block_start_N = (PID%num_blocks_along_N)*BLOCK_SIZE_N
    N_offset = (block_start_N + tl.arange(0, BLOCK_SIZE_N))
    N_mask = N_offset < N

    is_q = N_offset < N_q
    is_v = N_q + N_kv <= N_offset
    threshold = tl.full((BLOCK_SIZE_M,), threshold_k, dtype=tl.bfloat16)
    threshold = tl.where(is_q, threshold_q, threshold)
    threshold = tl.where(is_v, threshold_v, threshold)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        K_offset = (k*BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        K_mask = K_offset < K

        x_block = tl.load(x_ptr + M_offset[:, None]*x_M_stride + K_offset[None,:]*x_K_stride, mask=(M_mask[:, None] & K_mask[None,:]), other=0.0) # [BLOCK_SIZE_M, BLOCK_SIZE_K]
        idx = tl.abs(x_block.sum(axis=0)) > threshold # [BLOCK_SIZE_K]

        w_block = tl.load(w_ptr + N_offset[:, None]*w_N_stride + K_offset[None, :]*w_K_stride, mask=(N_mask[:, None] & (K_mask & idx)[None, :]), other=0.0)
        
        acc = tl.dot(x_block, w_block.T, acc=acc)

    tl.store(o_ptr + M_offset[:, None]*o_M_stride + N_offset[None, :]*o_N_stride, acc.to(tl.bfloat16), mask=(M_mask[:, None] & N_mask[None, :]))



def qkv_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold_q: float,
    threshold_k: float,
    threshold_v: float,
    sparsity_bin: int,
    kv_size: int
):
    N, K = weight.shape
    BS, SL, _ = x.shape
    assert x.shape[2] == K
    assert SL == 1
    x = x.view(-1, K)
    
    # assert weight.stride(1) > 1, "weight should be column major"

    output = torch.empty(BS*SL, N, device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(BS*SL, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
    qkv_gemm_kernel[grid](
        x,
        weight,
        output,
        threshold_q,
        threshold_k,
        threshold_v,
        BS*SL, N, K,
        N - 2*kv_size, kv_size,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BS
    )

    return output.reshape(BS, SL, -1)


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'kernels'))
from compile_wrapper import BaseKernel

# wrappers for compatibility with torch.compile
class SparseGEMV(BaseKernel):

    def meta(
        self,
        # The hidden states -> [ L, seq, D ].
        hidden_states: torch.Tensor,
        # weights [I, D]
        weights: torch.Tensor,
        threshold: float,
        sparsity_bin: int,
    ) -> torch.Tensor:
        return hidden_states.new_empty((hidden_states.size(0), hidden_states.size(1), weights.size(0)))

    def forward(
        self,
        # The hidden states -> [ L, seq, D ].
        hidden_states: torch.Tensor,
        # weights [I, D]
        weights: torch.Tensor,
        threshold: float,
        sparsity_bin: int,
    ) -> torch.Tensor:
        return splitk_sparse_gemv(hidden_states, weights, threshold, sparsity_bin) if hidden_states.shape[1] == 1 else splitk_sparse_gemm(hidden_states, weights, threshold, sparsity_bin)


from typing import Tuple
class SparseQKVGEMV(BaseKernel):
    def meta(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        sparsity_bin: int,
        kv_size: int
    ) -> torch.Tensor:
        return x.new_empty(x.shape[0], x.shape[1], weight.shape[0])

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        sparsity_bin: int,
        kv_size: int
    ) -> torch.Tensor:
        return qkv_gemv(x, weight, threshold_q, threshold_k, threshold_v, sparsity_bin, kv_size) if x.shape[1] == 1 else qkv_gemm(x, weight, threshold_q, threshold_k, threshold_v, sparsity_bin, kv_size)


# for testing purposes, to see if overhead at 0% is really due to strengthening torch.matmul (seems like it is)
class DenseGEMV(BaseKernel):

    def meta(self, x: torch.Tensor, W: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x.new_empty(x.shape[0], x.shape[1], W.shape[0])
    
    def forward(self, x: torch.Tensor, W: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.matmul(x, W.T)