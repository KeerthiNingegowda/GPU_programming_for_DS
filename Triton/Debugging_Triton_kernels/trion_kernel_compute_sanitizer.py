import triton
import triton.language as tl 
import torch


##Demoing compute_sanitizer debugging  mode -> Simple 1D tensor addition 
##Also demoing a reduction operation to demonstrate failure of race condition - A simple sum of elements in the array

#Buggy kernel - Add of 2 tensors
@triton.jit
def tensor_addition_1D_buggy(ip1_ptr, ip2_ptr, op_ptr, n_ele, BLOCK_SIZE:tl.constexpr):

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 99999999 ##Mess with the address 
    masks = offsets < n_ele

    ip1 = tl.load(ip1_ptr + offsets)  #BUG -  Remove mask
    ip2 = tl.load(ip2_ptr + offsets)  #BUG - Remove mask

    op = ip1 + ip2

    tl.store(op_ptr + offsets, op, mask=masks)

#Right Kernel - Add of 2 tensors
@triton.jit
def tensor_addition_1D(ip1_ptr, ip2_ptr, op_ptr, n_ele, BLOCK_SIZE:tl.constexpr):

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < n_ele

    ip1 = tl.load(ip1_ptr + offsets, mask=masks)
    ip2 = tl.load(ip2_ptr + offsets, mask=masks)

    op = ip1 + ip2

    tl.store(op_ptr + offsets, op, mask=masks)


#--------------Reduction Kernel---------------
##Sum of elements in a tensor 
#Easiest way to trigger race conditions are to remove atomic ops

#Right reduction kernel
@triton.jit
def sum_reduction_kernel_right(ip_ptr, n_ele, op_ptr, BLOCK_SIZE:tl.constexpr):

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    masks = offsets < n_ele

    ip = tl.load(ip_ptr+offsets, mask=masks)
    sum_num = tl.sum(ip)

    tl.atomic_add(op_ptr, sum_num)



#Buggy reduction kernel
@triton.jit
def sum_reduction_kernel_buggy(ip_ptr, n_ele, op_ptr, BLOCK_SIZE:tl.constexpr):

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    masks = offsets < n_ele

    ip = tl.load(ip_ptr+offsets, mask=masks)
    sum_num = tl.sum(ip)

    #Remove atomic sum and directly write sum_num as o/p
    tl.store(op_ptr, sum_num) 


def launch_sum_kernel():
    n_elements = 100

    X = torch.ones(n_elements, device="cuda", dtype=torch.float32)
    y = torch.zeros(1, device="cuda", dtype=torch.float32)

    grid = lambda meta : (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
    )

    #sum_reduction_kernel_right[grid](X, n_elements, y, BLOCK_SIZE=32)

    for _ in range(5):
        y.zero_()
        sum_reduction_kernel_buggy[grid](X, n_elements, y, BLOCK_SIZE=32)
        torch.cuda.synchronize()
        print(f"Sum is {y}") 


if __name__ == "__main__":
    #launch_1D_addition_kernel()
    #launch_sum_kernel()



def launch_1D_addition_kernel():
    n_elements = 100 ##To trigger initcheck - this shouldn't be a multiple of blocksize

    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    z = torch.zeros(n_elements, device="cuda", dtype=torch.float32)

    grid = lambda meta : (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
    )

    #tensor_addition_1D[grid](x, y, z, n_elements, BLOCK_SIZE=32)

    tensor_addition_1D_buggy[grid](x, y, z, n_elements, BLOCK_SIZE=32)


    torch.cuda.synchronize() ##If you dont add this the program will fail stating  target app failes before 1st API call

