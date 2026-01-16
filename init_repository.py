#!/usr/bin/env python3
"""Repository initialization script."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f" {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f" {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed: {e.stderr}")
        return False


def main():
    print("LLAMA-4 FINE-TUNING KIT - INITIALIZATION")
    print()
    
    if not Path("finetune_llama4_company.py").exists():
        print(" Error: Please run this script from the repository root")
        sys.exit(1)
    
    print("This script will:")
    print("1. Create necessary directories")
    print("2. Check Python version")
    print("3. Optionally initialize git repository")
    print("4. Optionally install dependencies")
    print()
    
    # Create directories
    dirs = ["datasets", "outputs", "logs"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    print(" Directories created")
    
    # Check Python version
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f" Python {version.major}.{version.minor} detected")
    else:
        print(f"  Python {version.major}.{version.minor} detected (3.10+ recommended)")
    
    # Ask about git
    print()
    response = input("Initialize git repository? (y/n): ").lower()
    if response == 'y':
        if run_command("git init", "Git initialization"):
            run_command("git add .", "Adding files to git")
            run_command('git commit -m "Initial commit: Complete Llama-4 fine-tuning kit"', "Creating initial commit")
    
    # Ask about dependencies
    print()
    response = input("Install dependencies now? (y/n): ").lower()
    if response == 'y':
        print("\nInstalling PyTorch (this may take a few minutes)...")
        run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "PyTorch installation"
        )
        
        print("\nInstalling other requirements...")
        run_command("pip install -r requirements.txt", "Requirements installation")
        
        print("\nInstalling Unsloth...")
        run_command(
            'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
            "Unsloth installation"
        )
        
        print("\n All dependencies installed!")
        print("\nRun environment check:")
        print("  python scripts/check_environment.py")
    
    print()
    print(" INITIALIZATION COMPLETE")
    print()
    print("Next steps:")
    print("1. Review README.md for detailed documentation")
    print("2. Check QUICKSTART.md for a 5-minute tutorial")
    print("3. Run: python scripts/check_environment.py")
    print("4. Prepare your dataset or use samples in data/")
    print("5. Start training: python finetune_llama4_company.py --help")
    print()


if __name__ == "__main__":
    main()
