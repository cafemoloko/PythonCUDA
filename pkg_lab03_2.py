# Zaprojektuj i zaimplementuj program doświadczalny, który wykaże, w jaki sposób odwijanie pętli 
# przeprowadzane przez kompilator NVCC wpływa na czas wykonania złożonych obliczeń. Przedyskutuj wyniki.

import datetime
from numba import cuda
import numpy as np 
import pragma

@cuda.jit()
@pragma.unroll()
def compute_gpu_unrolled(arr):
    for i in arr:
        arr[i] = i * np.pi ** np.pi

@cuda.jit()
def compute_gpu(arr):
    for i in arr:
        arr[i] = i * np.pi ** np.pi

def time_elapsed(fun):
    start = datetime.datetime.now()
    fun[blockspergrid, threadsperblock](arr)
    stop = datetime.datetime.now()
    elapsed = stop - start
    print(f'{elapsed}')

arr = np.arange(10)

#kernels
threadsperblock = 32
blockspergrid = (len(arr) + (threadsperblock - 1)) // threadsperblock
compute_gpu_unrolled[blockspergrid, threadsperblock](arr)

threadsperblock = 32
blockspergrid = (len(arr) + (threadsperblock - 1)) // threadsperblock
compute_gpu[blockspergrid, threadsperblock](arr)

# time measurements
print("Unrolled")
time_elapsed(compute_gpu_unrolled)
print("In loop")
time_elapsed(compute_gpu)
