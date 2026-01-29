import cupy as cp 
import numpy as np 

# print(f"Cupy version {cp.__version__}")
# print(f"Numpy version {np.__version__}")


x = cp.zeros((100,100), dtype=int) 
y = cp.ones((100,100), dtype=int)
z = x + y


a = np.zeros((100,100), dtype=int)
b = np.ones((100,100), dtype=int)
c = a + b

print(type(z))
print(type(c))
#z+c # This will err out as these are 2 different data types - You will have to move out either the cupy array to CPU or the other way round