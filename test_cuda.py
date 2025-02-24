import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA is not available. Checking why:")
    print("\nChecking NVIDIA driver:")
    try:
        import subprocess
        nvidia_smi = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(nvidia_smi.stdout.decode())
    except:
        print("nvidia-smi command failed - NVIDIA driver might not be installed")

# Test CUDA tensor operations
print("\nTrying CUDA tensor operations:")
try:
    x = torch.rand(5,3)
    if torch.cuda.is_available():
        x = x.cuda()
        print("Successfully created CUDA tensor")
        print(x)
    else:
        print("Created CPU tensor (CUDA not available)")
        print(x)
except Exception as e:
    print(f"Error during tensor operations: {str(e)}") 