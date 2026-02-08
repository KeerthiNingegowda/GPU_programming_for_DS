# GPU Kernel Programming for Data Scientists

This repository consists of exploration that helps understand how to write GPU kernels using Cupy, Numba and Triton. It also consists of writing CPU kernels using Numba's JIT compiler. Along with writing of GPU kernels, the repo also consists information on how to profile performance using Nvidia's Nsight Compute along with inspecting the corresponsing assembly language for fun 😊.  All on an AWS EC2 instance given everything is managed for you.

<b>Note:-</b> This exploration is applicable for optimization on a single GPU as opposed to GPU cluster.

Medium article can be found here:- TBD

## Starting point 🚨
You need to be extremely familiar with anatomy of a GPU at both compute and memory level and understand what are the bottlenecks of the same. If you are unfamiliar or not willing to put the effort then this entire excercise will <b>NOT</b> be a productive use of your time.
Make use of my notes as a staring point here:- https://github.com/KeerthiNingegowda/GPU_programming_for_DS/blob/main/GPU_CUDA_Basics%20-%20Google%20Docs.pdf

## Setting up EC2 instance and port forwarding to VS Code 
Check out this https://github.com/KeerthiNingegowda/GPU_programming_for_DS/blob/main/basic_housekeeping_stuff/AWS_EC2_Instance_Launch%20-%20Google%20Docs.pdf

## Cupy exploration
https://github.com/KeerthiNingegowda/GPU_programming_for_DS/tree/main/cupy_exploration 

## Numba exploration
CPU optimization - This matters especially when working in python due to
<ul>
<li>Python is an interpreter and is inherently slow. With Numba's njit you can achieve near C/C++ type speeds </li>
<li> If you data is not GPU worthy but still need some optimization this path may be your better bet</li>
<li>Not everyone has GPU access to begin with </li>
</ul>


Refer to ./Numba_exploration/njit_and_jit_exploration_CPU.ipynb.
GPU kernels
