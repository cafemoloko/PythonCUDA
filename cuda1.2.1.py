import numpy as np
import datetime

# Find_first() function returns a value of pi, when the difference 
# between consecutive elements in the list is smaller than
# given parameter (eps).
# The value of pi is calculated using Wallis product formula & numpy in 'arr'

def find_first(eps, arr):
    for i, x in enumerate(arr):
        if abs(x - arr[i-1]) < eps:
            yield x
            return

n = 100000
eps = 0.00001
a = np.arange(n)
arr = (4 * a ** 2) / (4 * a ** 2 - 1)

start = datetime.datetime.now()
ff = find_first(eps, arr)
for f in ff:
    print(f'Found value: {f}')
end = datetime.datetime.now()
elapsed_time = end - start
print(f'Done in: {elapsed_time}')
