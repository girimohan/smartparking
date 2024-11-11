import torch
import os

os.environ['CUDA_HOME'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6'
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Here's what PyTorch sees:")
    print(f"CUDA arch list: {torch.cuda.get_arch_list()}")
    print(f"CUDA compiler version: {torch.__config__.cuda_version}")

# Additional debug information
print(f"\nCUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"PATH: {os.environ.get('PATH', 'Not set')}")