from numba import cuda
import numpy as np
import datetime


def introduction():
    try:
        print(cuda.gpus)
        print(f'GPU name: {cuda.cudadrv.driver.Device(0).name}')
        print(f'GPU compute capability: {cuda.cudadrv.driver.Device(0).compute_capability}\n')
    except: 
        print(f'No CUDA-enabled GPU on your system.\n')

@cuda.jit
def on_gpu(arr):
    arr

#Compute function using GPU
@cuda.jit
def pow3(arr):
    for i in arr:
        arr[i] = arr[i] * arr[i] * arr[i] + arr[i] * arr[i] + arr[i]

#Compute function using CPU
def pow3_cpu(arr):
    for i in arr:
        arr[i] = arr[i] * arr[i] * arr[i] + arr[i] * arr[i] + arr[i]


introduction()
n = 1000
array = np.arange(n)
start = datetime.datetime.now()
pow3(array)
stop = datetime.datetime.now()
elapsed = stop - start
print(f'GPU done in: {elapsed} s')
print(f'First 5 results: {array[:5]}\n')

cpu_array = np.arange(n)
start = datetime.datetime.now()
pow3_cpu(cpu_array)
stop = datetime.datetime.now()
elapsed = stop - start
print(f'CPU done in: {elapsed} s')
print(f'First 5 results: {cpu_array[:5]}\n')

#Kernel
arr = np.arange(100000, dtype=np.float32)
threadsperblock = 32
blockspergrid = (arr.size + (threadsperblock - 1)) // threadsperblock
#measure time
start = datetime.datetime.now()
on_gpu[blockspergrid, threadsperblock](arr)
stop = datetime.datetime.now()
elapsed = stop - start
print(f'on_gpu done in: {elapsed} s')
