# script calculating the value of the polynomial of degree n (0 ≤ n < 128) on the CUDA device.

import numpy as np
from numba import cuda


@cuda.jit()
def polynomial(result, values):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    i = (bid * bdim) + tid
    cuda.atomic.add(result, 0, values[i])

x = 5
n = x ** np.arange(101) 
result = np.zeros(1)

polynomial[256,64](result, n)
print(result[0])
