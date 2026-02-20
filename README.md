# GPU Kernel Programming for Data Scientists

This repository consists of exploration that helps understand how to write GPU kernels using Cupy, Numba and Triton. It also consists writing CPU kernels in python using Numba's JIT compiler. Along with writing of GPU kernels, the repo also consists information on how to profile performance using Nvidia's Nsight Compute along with inspecting the corresponding assembly language for fun 😊.  All on an AWS EC2 instance given everything is managed for you.

<b>Note:-</b> This exploration is applicable for optimization on a single GPU as opposed to GPU cluster.

Medium article can be found here:- https://medium.com/@keerthi.ningegowda/in-a-parallel-universe-memory-is-scarce-a-peek-into-gpu-kernel-programming-and-debugging-29b27be20afc

## Starting point 🚨
You need to be extremely familiar with anatomy of a GPU at both compute and memory level and understand what are the bottlenecks of the same. If you are unfamiliar or not willing to put the effort then this entire excercise will <b>NOT</b> be a productive use of your time.
Make use of my notes as a staring point <a href="https://github.com/KeerthiNingegowda/GPU_programming_for_DS/blob/main/GPU_CUDA_Basics%20-%20Google%20Docs.pdf">here. </a>

## Setting up EC2 instance and port forwarding to VS Code 
Check out <a href="https://github.com/KeerthiNingegowda/GPU_programming_for_DS/blob/main/basic_housekeeping_stuff/AWS_EC2_Instance_Launch%20-%20Google%20Docs.pdf"> this </a> <br >
Note:- You probably need to request for vCPU access for gx series machines

## Cupy exploration
Notebooks can be found <a href="https://github.com/KeerthiNingegowda/GPU_programming_for_DS/tree/main/cupy_exploration"> here </a> <br >
Nsight compute results can be found <a href="https://github.com/KeerthiNingegowda/GPU_programming_for_DS/tree/main/cupy_exploration/ncc_kernel_performance_analysis"> here </a>

## Numba exploration
CPU optimization - This matters especially when working in python due to
<ul>
<li>Python uses an interpreter and is inherently slow. With Numba's njit you can achieve near C/C++ type speeds</li>
<li> If your data is not GPU worthy but still need some optimization this path may be your better bet</li>
<li>Not everyone has GPU access to begin with </li>
</ul>

Refer to [./Numba_exploration/njit_and_jit_exploration_CPU.ipynb.](https://github.com/KeerthiNingegowda/GPU_programming_for_DS/blob/main/Numba_exploration/njit_and_jit_exploration_CPU.ipynb)

GPU kernels -
Refer to notebooks <a href = "https://github.com/KeerthiNingegowda/GPU_programming_for_DS/blob/main/Numba_exploration/numba_cuda_jit_exploration_GPU.ipynb"> here </a> and <a href = "https://github.com/KeerthiNingegowda/GPU_programming_for_DS/blob/main/Numba_exploration/numba_cudajit_custom_kernel.ipynb"> here </a>.

Check out PTX analysis <a href = "https://github.com/KeerthiNingegowda/GPU_programming_for_DS/blob/main/Numba_exploration/PTX_FMA_Numba.pdf"> here </a>


## Triton exploration
Kernels can be found <a href="./Triton/triton_custom_kernels.ipynb"> here </a> and <a href="./Triton/Triton_custom_kernels_2D.ipynb"> here(This is for 2D tensors) </a>

Additionally, debugging example scripts can be found <a href="./Triton/Debugging_Triton_kernels/"> here </a>

1) Easiest mode - Triton Intepreter mode

```
export TRITON_INTERPRET=1
```

Set the above env variable to inspect for any logical issues with your kernels

2) Use compute-sanitizer from nvidia to check any memory access related issues

Command to run memcheck (Is memory address valid?)
```
compute-sanitizer --tool memcheck python <filename.py>
```

Other tools you can use initcheck and racecheck (I couldnt trigger race check. But it may work for you)

Some of the evidence of related to debugging Triton kernels can be found <a href="./Triton/Debugging_Triton_kernels/debug_evidence/"> here </a>


## Other basic commands
Running Nsight Compute profiling report
```
ncu python <kernel-file-name.py>
```
If you have multiple kernels in one file
```
ncu --kernel-name <kernel_name> python <file-name.py>
```
When working with Triton you may likely see errors related to python development headers - Something like #include <Python.h> not found. Install it
```
sudo yum install python3-devel  # This varies based on your OS
```
