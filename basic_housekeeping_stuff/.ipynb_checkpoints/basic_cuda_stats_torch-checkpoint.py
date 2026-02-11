import torch

## Answer to all these questions using torch.cuda - torch because usually it is quite widely used

#1
print(f"Is cuda available? {torch.cuda.is_available()}")

#2
print(f"How many GPUs do I have acess to? {torch.cuda.device_count()}")

#3
print(f"Which GPU device am I using? {torch.cuda.get_device_name(0)}")

#4
print(f"Which cuda version is pytorch talking with? {torch.version.cuda}")

#5
print(f"Which device is currently active? {torch.cuda.current_device()}")

#6
print(f"How much memory is allocated? {torch.cuda.memory_allocated()}")

#7
print(f"Memory usage summary {torch.cuda.memory_summary()}")

#8
print(f"Device capability  {torch.cuda.get_device_capability()}")

#9
print(f"Does this GPU support mixed precision {torch.cuda.is_bf16_supported()}")


#torch.cuda.set_device(<d>) ##This should match how many devices you have when you ran torch.cuda.device_count()

#Freeing unused cache memory  torch.cuda.empty_cache()

#IMP - Synchronize CPU and GPU before timing a process on GPU - GPU executes lazily. The jobs are queued on the GPU by CPU 
#and until the result is needed it wont be exceuted until the result is actually needed. So this becomes important if you are timing GPU
#torch.cuda.synchronize()
