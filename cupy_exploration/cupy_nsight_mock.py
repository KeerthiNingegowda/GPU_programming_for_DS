import cupy as cp
import numpy as np
import time

SIZE = 1e8

input_array = cp.random.uniform(low = -5, high = 5, size=int(SIZE))
print(f"Min and max of the input array {input_array.min()} {input_array.max()}")
mu = cp.mean(input_array)
sigma = cp.std(input_array)


normalize_and_clip  = cp.ElementwiseKernel(
    in_params = "float64 in_array, float64 mu, float64 sigma",
    out_params = "float64 output_y",
    operation = """
    float z = (in_array - mu) / sigma;
    if (z > 3.0f) {                     
       output_y = 3.0f; }
    else if (z < -3.0f) {
        output_y = -3.0f; }
    else {
        output_y = z;
    }
    """,
    name = "normalize_and_clip")


cp.cuda.Device().synchronize()
start_1 = time.perf_counter()

k = normalize_and_clip(input_array, mu, sigma)
cp.cuda.Device().synchronize()

print(f"Total time for execution via Kernel {time.perf_counter() - start_1}")
print(f"Min and Max are as follows {k.min(), k.max()}")