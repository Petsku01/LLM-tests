FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Copy project files
COPY . .

# Install package
RUN pip3 install -e .

# Set entrypoint
ENTRYPOINT ["python3", "finetune_llama4_company.py"]
CMD ["--help"]
