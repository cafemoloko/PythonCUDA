from numba import cuda
import numpy as np
import datetime


def introduction():
    try:
        print(cuda.gpus)
        print(f'GPU name: {cuda.cudadrv.driver.Device(0).name}')
        print(f'GPU compute capability: {cuda.cudadrv.driver.Device(0).compute_capability}')
    except: 
        print(f"No CUDA-enabled GPU on your system.\n"
              f"Type 'export NUMBA_ENABLE_CUDASIM=1' on Linux terminal or\n"
              f"'SET NUMBA_ENABLE_CUDASIM=1' on Windows cmd shell\n"
              f"to enable Numba CUDA simulator")
#GPU
@cuda.jit
def pow3(arr):
    for i in arr:
        arr[i] = arr[i] * arr[i] * arr[i] + arr[i] * arr[i] + arr[i]

#CPU
def pow3_cpu(arr):
    for i in arr:
        arr[i] = arr[i] * arr[i] * arr[i] + arr[i] * arr[i] + arr[i]

def main():
    introduction()
    n = 1000
    array = np.arange(n)
    start = datetime.datetime.now()
    pow3(array)
    stop = datetime.datetime.now()
    elapsed = stop - start
    print(f'GPU done in: {elapsed} s')
    print(f'First 5 results: {array[:5]}')

    cpu_array = np.arange(n)
    start = datetime.datetime.now()
    pow3_cpu(cpu_array)
    stop = datetime.datetime.now()
    elapsed = stop - start
    print(f'CPU done in: {elapsed} s')
    print(f'First 5 results: {cpu_array[:5]}')
 
if __name__ == '__main__':
    main()
