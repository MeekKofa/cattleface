import torch
import subprocess
import sys
import os

def check_nvidia_smi():
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi'], stderr=subprocess.PIPE)
        return True, nvidia_smi.decode()
    except:
        return False, "nvidia-smi not found or not accessible"

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    
    # Check CUDA build info
    print("\nPyTorch CUDA build info:")
    print("CUDA build version:", torch.version.cuda)
    
    # Check if NVIDIA driver is installed
    has_nvidia, nvidia_output = check_nvidia_smi()
    print("\nNVIDIA Driver check:")
    if has_nvidia:
        print("NVIDIA driver is installed")
        print("\nnvidia-smi output:")
        print(nvidia_output)
    else:
        print("NVIDIA driver is not installed or not accessible")
    
    # If CUDA is available, print detailed information
    if torch.cuda.is_available():
        print("\nCUDA Device Information:")
        print("Current CUDA device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name())
        print("Device count:", torch.cuda.device_count())
        print("Device capabilities:", torch.cuda.get_device_capability())
        print("Memory allocated:", torch.cuda.memory_allocated())
        print("Memory cached:", torch.cuda.memory_cached())
    else:
        print("\nPossible issues:")
        print("1. NVIDIA driver not installed")
        print("2. CUDA toolkit not installed")
        print("3. PyTorch not installed with CUDA support")
        print("\nTo fix:")
        print("1. Install NVIDIA drivers from https://www.nvidia.com/download/index.aspx")
        print("2. Reinstall PyTorch with CUDA support using:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    main()
