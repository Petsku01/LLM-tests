"""Setup script for Llama-4 Fine-Tuning Kit"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="llama4-finetuning-kit",
    version="1.0.0",
    description="Fine-tuning pipeline for Llama-4 models using Unsloth + QLoRA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/llama4-finetuning-kit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.45.0,<=4.57.3",
        "accelerate>=0.34.0",
        "peft>=0.12.0,<=0.18.1",
        "trl>=0.11.0,<=0.24.0",
        "bitsandbytes>=0.44.0",
        "datasets>=2.20.0,<=4.5.0",
        "sentencepiece>=0.2.0",
        "tqdm>=4.66.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.1",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "black>=24.4.0",
            "flake8>=7.1.0",
            "mypy>=1.10.0",
            "pre-commit>=3.7.0",
        ],
        "monitoring": [
            "wandb>=0.17.0",
            "tensorboard>=2.17.0",
        ],
        "evaluation": [
            "evaluate>=0.4.2",
            "rouge-score>=0.1.2",
            "sacrebleu>=2.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llama4-finetune=finetune_llama4_company:main",
        ],
    },
)
