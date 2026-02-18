import triton
import triton.language as tl 
import torch


##Demoing triton interpreter mode -> Simple 1D tensor addition

@triton.jit
def tensor_addition_1D(ip1_ptr, ip2_ptr, op_ptr, n_ele, BLOCK_SIZE:tl.constexpr):

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < n_ele

    #Print masks and offsets and pid
    tl.device_print("pid", pid)
    tl.device_print("offsets", offsets)
    tl.device_print("masks", masks)
    
    ##Static asserts - useful 
    tl.static_assert(BLOCK_SIZE<=32)

    ip1 = tl.load(ip1_ptr + offsets, mask=masks)
    ip2 = tl.load(ip2_ptr + offsets, mask=masks)

    #print the loaded ips 
    tl.device_print("IP1", ip1)

    op = ip1 + ip2

    tl.store(op_ptr + offsets, op, mask=masks)
    #honestly redundant - you can just print out masks at block level
    #tl.store(op_ptr + offsets, masks.to(tl.float32), mask=masks)


def launch_1D_addition_kernel():
    n_elements = 10

    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    z = torch.zeros(n_elements, device="cuda", dtype=torch.float32)

    grid = lambda meta : (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
    )


    tensor_addition_1D[grid](x, y, z, n_elements, BLOCK_SIZE=32)

    # print(z)


if __name__ == "__main__":
    launch_1D_addition_kernel()
