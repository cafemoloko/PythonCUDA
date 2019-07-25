# Zaprojektuj i zaimplementuj program doświadczalny, który wykaże, w jaki sposób arytmetyka 
# zmiennoprzecinkowa urządzeń CUDA wpływa na dokładność złożonych obliczeń. Porównaj wyniki 
# uzyskane z urządzenia i z hosta i przedyskutuj je.


from numba import cuda
import numpy as np 


@cuda.jit()
def compute_gpu(arr):
    for i in arr:
        arr[i] = i * np.pi ** np.pi

arr = np.arange(10)
input_exp = [n * np.pi ** np.pi for n in arr]
print(f'CPU: {input_exp}')

#kernel
threadsperblock = 32
blockspergrid = (len(arr) + (threadsperblock - 1)) // threadsperblock
compute_gpu[blockspergrid, threadsperblock](arr)
print(f'GPU: {arr}')