"""Smoke test: reference solutions for all 14 GPU-Puzzles under NUMBA's CUDA simulator.

NOTE: The puzzles' `CudaProblem.check()` wraps kernels as closures that capture
`cuda` as a parameter. Under NUMBA_ENABLE_CUDASIM=1 the simulator only swaps
`cuda.*` references that live in the function's __globals__, so the closure
pattern bypasses the swap. This file defines kernels at module scope where
`cuda` is a real global, launches them manually, and compares to the puzzle spec.
"""
import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import sys
import warnings
import numba
import numpy as np
from numba import cuda

warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning, module="numba")


def check(name, got, want):
    try:
        np.testing.assert_allclose(got, want)
        print(f"  [OK]   {name}")
        return True
    except AssertionError as e:
        print(f"  [FAIL] {name}: {str(e).splitlines()[0]}")
        return False


# -------- specs --------
def map_spec(a): return a + 10
def zip_spec(a, b): return a + b

def pool_spec(a):
    out = np.zeros(*a.shape)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0): i + 1].sum()
    return out

def dot_spec(a, b): return a @ b

def conv_spec(a, b):
    out = np.zeros(*a.shape)
    n = b.shape[0]
    for i in range(a.shape[0]):
        out[i] = sum(a[i + j] * b[j] for j in range(n) if i + j < a.shape[0])
    return out

def sum_block_spec(a, tpb):
    out = np.zeros((a.shape[0] + tpb - 1) // tpb)
    for j, i in enumerate(range(0, a.shape[-1], tpb)):
        out[j] = a[i: i + tpb].sum()
    return out

def axis_sum_spec(a, tpb):
    out = np.zeros((a.shape[0], (a.shape[1] + tpb - 1) // tpb))
    for j, i in enumerate(range(0, a.shape[-1], tpb)):
        out[..., j] = a[..., i: i + tpb].sum(-1)
    return out

def matmul_spec(a, b): return a @ b


# -------- kernels (top-level, so `cuda` is a global the simulator can swap) --------

@cuda.jit
def k_map(out, a):
    i = cuda.threadIdx.x
    out[i] = a[i] + 10

@cuda.jit
def k_zip(out, a, b):
    i = cuda.threadIdx.x
    out[i] = a[i] + b[i]

@cuda.jit
def k_guard(out, a, size):
    i = cuda.threadIdx.x
    if i < size:
        out[i] = a[i] + 10

@cuda.jit
def k_map2d(out, a, size):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    if i < size and j < size:
        out[i, j] = a[i, j] + 10

@cuda.jit
def k_broadcast(out, a, b, size):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    if i < size and j < size:
        out[i, j] = a[i, 0] + b[0, j]

@cuda.jit
def k_blocks(out, a, size):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < size:
        out[i] = a[i] + 10

@cuda.jit
def k_blocks2d(out, a, size):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < size and j < size:
        out[i, j] = a[i, j] + 10

TPB_SHARED = 4
@cuda.jit
def k_shared(out, a, size):
    shared = cuda.shared.array(TPB_SHARED, numba.float32)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x
    if i < size:
        shared[local_i] = a[i]
        cuda.syncthreads()
        out[i] = shared[local_i] + 10

TPB_POOL = 8
@cuda.jit
def k_pool(out, a, size):
    shared = cuda.shared.array(TPB_POOL, numba.float32)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x
    if i < size:
        shared[local_i] = a[i]
    cuda.syncthreads()
    if i < size:
        total = 0.0
        for k in range(3):
            if local_i - k >= 0:
                total += shared[local_i - k]
        out[i] = total

TPB_DOT = 8
@cuda.jit
def k_dot(out, a, b, size):
    shared = cuda.shared.array(TPB_DOT, numba.float32)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x
    if i < size:
        shared[local_i] = a[i] * b[i]
    cuda.syncthreads()
    if local_i == 0:
        total = 0.0
        for k in range(TPB_DOT):
            total += shared[k]
        out[0] = total

TPB_CONV = 8
MAX_CONV = 4
TPB_MAX_CONV = TPB_CONV + MAX_CONV
@cuda.jit
def k_conv(out, a, b, a_size, b_size):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x

    shared_a = cuda.shared.array(TPB_MAX_CONV, numba.float32)
    shared_b = cuda.shared.array(MAX_CONV, numba.float32)

    if i < a_size:
        shared_a[local_i] = a[i]
    if local_i < b_size - 1:
        halo_idx = i + TPB_CONV
        if halo_idx < a_size:
            shared_a[local_i + TPB_CONV] = a[halo_idx]
    if local_i < b_size:
        shared_b[local_i] = b[local_i]
    cuda.syncthreads()

    if i < a_size:
        total = 0.0
        for j in range(b_size):
            if i + j < a_size:
                total += shared_a[local_i + j] * shared_b[j]
        out[i] = total

TPB_SUM = 8
@cuda.jit
def k_sum(out, a, size):
    cache = cuda.shared.array(TPB_SUM, numba.float32)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x
    if i < size:
        cache[local_i] = a[i]
    else:
        cache[local_i] = 0
    cuda.syncthreads()
    k = TPB_SUM // 2
    while k > 0:
        if local_i < k:
            cache[local_i] += cache[local_i + k]
        cuda.syncthreads()
        k //= 2
    if local_i == 0:
        out[cuda.blockIdx.x] = cache[0]

TPB_AXIS = 8
@cuda.jit
def k_axis_sum(out, a, size):
    cache = cuda.shared.array(TPB_AXIS, numba.float32)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x
    batch = cuda.blockIdx.y
    if i < size:
        cache[local_i] = a[batch, i]
    else:
        cache[local_i] = 0
    cuda.syncthreads()
    k = TPB_AXIS // 2
    while k > 0:
        if local_i < k:
            cache[local_i] += cache[local_i + k]
        cuda.syncthreads()
        k //= 2
    if local_i == 0:
        out[batch, cuda.blockIdx.x] = cache[0]

TPB_MM = 3
@cuda.jit
def k_matmul(out, a, b, size):
    a_shared = cuda.shared.array((TPB_MM, TPB_MM), numba.float32)
    b_shared = cuda.shared.array((TPB_MM, TPB_MM), numba.float32)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y

    total = 0.0
    for tile in range(0, size, TPB_MM):
        if i < size and tile + local_j < size:
            a_shared[local_i, local_j] = a[i, tile + local_j]
        else:
            a_shared[local_i, local_j] = 0
        if j < size and tile + local_i < size:
            b_shared[local_i, local_j] = b[tile + local_i, j]
        else:
            b_shared[local_i, local_j] = 0
        cuda.syncthreads()
        for k in range(TPB_MM):
            total += a_shared[local_i, k] * b_shared[k, local_j]
        cuda.syncthreads()

    if i < size and j < size:
        out[i, j] = total


# -------- driver --------
print("Running smoke tests for all 14 GPU-Puzzles under NUMBA_ENABLE_CUDASIM=1\n")
passed = 0
total = 0

# 1
SIZE = 4
a = np.arange(SIZE); out = np.zeros(SIZE)
k_map[(1,), (SIZE,)](out, a)
total += 1; passed += check("01 Map", out, map_spec(a))

# 2
a = np.arange(SIZE); b = np.arange(SIZE); out = np.zeros(SIZE)
k_zip[(1,), (SIZE,)](out, a, b)
total += 1; passed += check("02 Zip", out, zip_spec(a, b))

# 3
a = np.arange(SIZE); out = np.zeros(SIZE)
k_guard[(1,), (8,)](out, a, SIZE)
total += 1; passed += check("03 Guard", out, map_spec(a))

# 4
SIZE = 2
a = np.arange(SIZE * SIZE).reshape(SIZE, SIZE); out = np.zeros((SIZE, SIZE))
k_map2d[(1,), (3, 3)](out, a, SIZE)
total += 1; passed += check("04 Map 2D", out, map_spec(a))

# 5
a = np.arange(SIZE).reshape(SIZE, 1); b = np.arange(SIZE).reshape(1, SIZE); out = np.zeros((SIZE, SIZE))
k_broadcast[(1,), (3, 3)](out, a, b, SIZE)
total += 1; passed += check("05 Broadcast", out, zip_spec(a, b))

# 6
SIZE = 9
a = np.arange(SIZE); out = np.zeros(SIZE)
k_blocks[(3,), (4,)](out, a, SIZE)
total += 1; passed += check("06 Blocks", out, map_spec(a))

# 7
SIZE = 5
a = np.ones((SIZE, SIZE)); out = np.zeros((SIZE, SIZE))
k_blocks2d[(2, 2), (3, 3)](out, a, SIZE)
total += 1; passed += check("07 Blocks 2D", out, map_spec(a))

# 8
SIZE = 8
a = np.ones(SIZE); out = np.zeros(SIZE)
k_shared[(2,), (TPB_SHARED,)](out, a, SIZE)
total += 1; passed += check("08 Shared", out, map_spec(a))

# 9
SIZE = 8
a = np.arange(SIZE); out = np.zeros(SIZE)
k_pool[(1,), (TPB_POOL,)](out, a, SIZE)
total += 1; passed += check("09 Pooling", out, pool_spec(a))

# 10
SIZE = 8
a = np.arange(SIZE); b = np.arange(SIZE); out = np.zeros(1)
k_dot[(1,), (SIZE,)](out, a, b, SIZE)
total += 1; passed += check("10 Dot", out, np.array([dot_spec(a, b)]))

# 11a
a = np.arange(6); b = np.arange(3); out = np.zeros(6)
k_conv[(1,), (TPB_CONV,)](out, a, b, 6, 3)
total += 1; passed += check("11 Conv (Simple)", out, conv_spec(a, b))

# 11b
a = np.arange(15); b = np.arange(4); out = np.zeros(15)
k_conv[(2,), (TPB_CONV,)](out, a, b, 15, 4)
total += 1; passed += check("11 Conv (Full)", out, conv_spec(a, b))

# 12a
SIZE = 8
a = np.arange(SIZE); out = np.zeros(1)
k_sum[(1,), (TPB_SUM,)](out, a, SIZE)
total += 1; passed += check("12 Sum (Simple)", out, sum_block_spec(a, TPB_SUM))

# 12b
SIZE = 15
a = np.arange(SIZE); out = np.zeros(2)
k_sum[(2,), (TPB_SUM,)](out, a, SIZE)
total += 1; passed += check("12 Sum (Full)", out, sum_block_spec(a, TPB_SUM))

# 13
BATCH, SIZE = 4, 6
a = np.arange(BATCH * SIZE).reshape(BATCH, SIZE); out = np.zeros((BATCH, 1))
k_axis_sum[(1, BATCH), (TPB_AXIS, 1)](out, a, SIZE)
total += 1; passed += check("13 Axis Sum", out, axis_sum_spec(a, TPB_AXIS))

# 14a
SIZE = 2
inp1 = np.arange(SIZE * SIZE).reshape(SIZE, SIZE)
inp2 = np.arange(SIZE * SIZE).reshape(SIZE, SIZE).T
out = np.zeros((SIZE, SIZE))
k_matmul[(1, 1), (TPB_MM, TPB_MM)](out, inp1, inp2, SIZE)
total += 1; passed += check("14 Matmul (Simple)", out, matmul_spec(inp1, inp2))

# 14b
SIZE = 8
inp1 = np.arange(SIZE * SIZE).reshape(SIZE, SIZE)
inp2 = np.arange(SIZE * SIZE).reshape(SIZE, SIZE).T
out = np.zeros((SIZE, SIZE))
k_matmul[(3, 3), (TPB_MM, TPB_MM)](out, inp1, inp2, SIZE)
total += 1; passed += check("14 Matmul (Full)", out, matmul_spec(inp1, inp2))

print(f"\n{passed}/{total} passed")
sys.exit(0 if passed == total else 1)
