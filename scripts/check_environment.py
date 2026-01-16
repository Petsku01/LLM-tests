#!/usr/bin/env python3
"""Check environment setup for fine-tuning."""

import sys
import subprocess


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(" Python 3.10+ required")
        return False
    else:
        print(" Python version OK")
        return True


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f" CUDA available: {torch.version.cuda}")
            print(f" GPU: {torch.cuda.get_device_name(0)}")
            print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("  CUDA not available (CPU mode)")
            return False
    except ImportError:
        print(" PyTorch not installed")
        return False


def check_package(package_name: str, import_name: str = None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f" {package_name}: {version}")
        return True
    except ImportError:
        print(f" {package_name} not installed")
        return False


def main():
    """Main checker."""
    print("ENVIRONMENT VALIDATION")
    print()
    
    checks = []
    
    # Python version
    checks.append(check_python_version())
    print()
    
    # CUDA
    checks.append(check_cuda())
    print()
    
    # Required packages
    print("Checking required packages...")
    packages = [
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("bitsandbytes", "bitsandbytes"),
        ("datasets", "datasets"),
        ("unsloth", "unsloth"),
    ]
    
    for pkg_name, import_name in packages:
        checks.append(check_package(pkg_name, import_name))
    
    print()
    
    if all(checks[1:]):  # Skip CUDA check (warning only)
        print(" Environment setup complete!")
    else:
        print(" Some packages are missing. Install with:")
        print("   pip install -r requirements.txt")
        print("   pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
    

if __name__ == "__main__":
    main()
