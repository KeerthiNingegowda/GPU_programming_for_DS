# GPU_programming_for_DS


Fundamentals about the hardware i.e. GPU is important to understand before going ahead with programming kernels or even just using pytorch wrappers to efficiently utilize GPU 

Key concepts:-
Kernels, threads, warps, blocks, grids
Memory hierarchies in a GPU - Registers, Shared memory, L1 cache, L2 cache, Global Memory(VRAM), Host(CPU) memory
Learn about Warp divergence, memory coalesce, kernel fusion - These are not fancy words but fundamentals.

It is also important to know when not to use GPUs'. They are very elegant in data processing(including model training) based on the assumption that the data can be accessed in blocks of contiguous memory blocks and that the computation is embarrassingly parallel  i.e. the results of one wont depend on the other. So when this pattern is broken, GPUs struggle. Instances where you trying to solve some graph problem or accessing a data structure like linked list where you cannot access is "not direct" is when GPUs struggle.

Understand key questions like
1) Can different threads on the same block execute different kernels?
2) How does the CPU-GPU memory transfer happen? What are some caveats of the same?
    a) What is Unified memory architecture between CPU and GPU
3) What are tensors and how are they different from matrices? Do tensors always have to be matrices?
4) What is the difference between a tensor core and a cuda core


In any scenario, when using GPU you will either be bounded my compute (how many instructions can I crunch) or by memory (data in GPU memory, time spent in data transfers between GPU and CPU etc,.)