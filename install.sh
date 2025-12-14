#!/bin/bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if uv is installed
print_info "Checking if uv is installed..."
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
print_success "uv is installed"

# Check if git submodules are initialized
print_info "Checking git submodules..."
if [ ! -d "submodules/ultralytics" ] || [ ! -f "submodules/ultralytics/.git" ]; then
    print_info "Initializing git submodules..."
    git submodule update --init --recursive
fi
print_success "Git submodules are ready"

# Detect CUDA version
print_info "Detecting CUDA version..."
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' || echo "")
fi

if [ -z "$CUDA_VERSION" ]; then
    # Try to infer from nvidia-smi driver version
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo "")
        if [ ! -z "$DRIVER_VERSION" ]; then
            if [ "$DRIVER_VERSION" -ge 580 ]; then
                CUDA_VERSION="12.1"
            elif [ "$DRIVER_VERSION" -ge 550 ]; then
                CUDA_VERSION="12.0"
            elif [ "$DRIVER_VERSION" -ge 535 ]; then
                CUDA_VERSION="11.8"
            else
                CUDA_VERSION="11.8"
            fi
        fi
    fi
fi

# Default to cu121 if detection fails
if [ -z "$CUDA_VERSION" ]; then
    print_warning "Could not detect CUDA version, defaulting to CUDA 12.1"
    CUDA_VERSION="12.1"
    PYTORCH_CUDA="cu121"
elif [ "$(echo "$CUDA_VERSION >= 12.1" | bc -l 2>/dev/null || echo "0")" = "1" ]; then
    PYTORCH_CUDA="cu121"
elif [ "$(echo "$CUDA_VERSION >= 12.0" | bc -l 2>/dev/null || echo "0")" = "1" ]; then
    PYTORCH_CUDA="cu121"  # cu121 is backward compatible with 12.0
elif [ "$(echo "$CUDA_VERSION >= 11.8" | bc -l 2>/dev/null || echo "0")" = "1" ]; then
    PYTORCH_CUDA="cu118"
else
    # For older CUDA versions, use cu118 which is backward compatible
    print_warning "CUDA version $CUDA_VERSION detected. Using cu118 (backward compatible)"
    PYTORCH_CUDA="cu118"
fi

print_success "Detected CUDA version: $CUDA_VERSION, using PyTorch with $PYTORCH_CUDA"

# Check GPU availability
print_info "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        print_success "Found $GPU_COUNT GPU(s):"
        nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>/dev/null | while IFS=, read -r idx name driver; do
            echo "  GPU $idx: $name (Driver: $driver)"
        done
    else
        print_warning "No GPUs detected"
    fi
else
    print_warning "nvidia-smi not available, cannot verify GPU"
fi

# Create virtual environment using uv
print_info "Creating virtual environment with uv..."
if [ -d ".venv" ]; then
    print_warning ".venv already exists, skipping creation"
else
    uv venv .venv
    print_success "Virtual environment created at .venv"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip in the venv
print_info "Upgrading pip..."
uv pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
print_info "Installing PyTorch with CUDA support ($PYTORCH_CUDA)..."
uv pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA"
print_success "PyTorch with CUDA installed"

# Install ultralytics from submodule in editable mode
print_info "Installing ultralytics from submodule (editable mode)..."
uv pip install -e submodules/ultralytics
print_success "Ultralytics installed from submodule"

# Verify GPU availability with PyTorch
print_info "Verifying GPU availability with PyTorch..."
python3 << EOF
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print("\n✓ GPU support is working correctly!")
    sys.exit(0)
else:
    print("\n✗ GPU support is not available")
    print("  This may be due to:")
    print("  - Missing NVIDIA drivers")
    print("  - CUDA toolkit not installed")
    print("  - PyTorch CUDA version mismatch")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "GPU verification passed!"
else
    print_error "GPU verification failed. Please check your CUDA installation."
    exit 1
fi

print_success "Installation completed successfully!"
print_info "To activate the virtual environment in the future, run:"
echo "  source .venv/bin/activate"

