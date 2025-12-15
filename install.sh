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

# Check if git submodules are initialized and checked out
print_info "Checking git submodules..."
SUBMODULES_NEED_UPDATE=false
for submodule in submodules/ultralytics submodules/sam3 submodules/inference submodules/supervision; do
    if [ ! -d "$submodule" ] || [ ! -f "$submodule/.git" ] || [ -z "$(ls -A "$submodule" 2>/dev/null)" ]; then
        SUBMODULES_NEED_UPDATE=true
        break
    fi
done

if [ "$SUBMODULES_NEED_UPDATE" = true ]; then
    print_info "Initializing and updating git submodules (shallow clone, latest commit only)..."
    git submodule update --init --recursive --depth 1
    # Ensure submodules are checked out (sometimes they're initialized but not checked out)
    for submodule in submodules/ultralytics submodules/sam3 submodules/inference submodules/supervision; do
        if [ -f "$submodule/.git" ] && [ -z "$(ls -A "$submodule" 2>/dev/null)" ]; then
            print_info "Checking out $submodule..."
            git -C "$submodule" checkout -f HEAD 2>/dev/null || true
        fi
    done
fi
print_success "Git submodules are ready"

# Detect platform
PLATFORM=$(uname -s)
IS_MACOS=false
if [ "$PLATFORM" = "Darwin" ]; then
    IS_MACOS=true
fi

# Detect CUDA version (only on Linux)
CUDA_VERSION=""
PYTORCH_CUDA=""
if [ "$IS_MACOS" = false ]; then
    print_info "Detecting CUDA version..."
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
else
    print_info "macOS detected - will install PyTorch with MPS (Metal Performance Shaders) support"
fi

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

# Determine and install correct Python version using uv
print_info "Checking Python version requirements..."
REQUIRED_PYTHON=">=3.9,<3.13"
print_info "Project requires Python $REQUIRED_PYTHON (numpy 1.26 requires >=3.9,<3.13)"

# Check current Python version (macOS-compatible)
CURRENT_PYTHON_VERSION=$(python3 --version 2>&1 | sed -E 's/^Python ([0-9]+\.[0-9]+).*/\1/' | head -1)
print_info "Current Python version: $CURRENT_PYTHON_VERSION"

# Use uv to install/ensure correct Python version (3.12 recommended)
PYTHON_VERSION="3.12"
print_info "Installing Python $PYTHON_VERSION using uv (if not already available)..."
uv python install "$PYTHON_VERSION" 2>/dev/null || print_warning "Python $PYTHON_VERSION may already be installed or installation skipped"
print_success "Python $PYTHON_VERSION ready"

# Create virtual environment using uv with specific Python version
print_info "Creating virtual environment with uv using Python $PYTHON_VERSION..."
if [ -d ".venv" ]; then
    print_warning ".venv already exists, removing it to recreate with correct Python version..."
    rm -rf .venv
fi
uv venv .venv --python "$PYTHON_VERSION"
print_success "Virtual environment created at .venv with Python $PYTHON_VERSION"

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

# Verify submodules exist before syncing
print_info "Verifying submodules..."
if [ ! -d "submodules/ultralytics" ] || [ ! -d "submodules/sam3" ]; then
    print_error "Submodules not found. Please ensure git submodules are initialized."
    exit 1
fi
print_success "Submodules verified"

# Install PyTorch (CUDA on Linux, MPS on macOS)
# Check if PyTorch is already installed to skip reinstallation
if python3 -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q .; then
    INSTALLED_TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    print_info "PyTorch already installed (version: $INSTALLED_TORCH_VERSION), skipping installation"
    print_info "If you need to reinstall, remove .venv and run the script again"
else
    if [ "$IS_MACOS" = true ]; then
        print_info "Installing PyTorch with MPS (Metal Performance Shaders) support for macOS..."
        print_info "This may take a few minutes as PyTorch packages are large (~2-3 GB download)..."
        uv pip install torch torchvision torchaudio
        print_success "PyTorch with MPS support installed"
    else
        print_info "Installing PyTorch with CUDA support ($PYTORCH_CUDA)..."
        print_info "This may take a few minutes as PyTorch packages are large (~2-3 GB download)..."
        # uv shows progress by default, no need for --progress-bar flag
        uv pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA"
        print_success "PyTorch with CUDA installed"
    fi
fi

# Setup .env file for configuration and secrets
print_info "Setting up .env file for configuration..."
if [ ! -f ".env" ]; then
    print_info "Creating .env file template..."
    cat > .env << 'EOF'
# Hugging Face Configuration
# Get your token from: https://huggingface.co/settings/tokens
# Request access to SAM3 model at: https://huggingface.co/facebook/sam3
HF_TOKEN=your_huggingface_token_here

# SAM3 Model Configuration
SAM3_MODEL_PATH=models/sam3.pt

# SAM3 BPE Vocabulary Path
# Path to the BPE vocabulary file (default: models/bpe_simple_vocab_16e6.txt.gz)
# If left empty, will use the package default location
SAM3_BPE_PATH=models/bpe_simple_vocab_16e6.txt.gz
EOF
    print_warning ".env file created. Please edit it and add your Hugging Face token."
    print_warning "You need to:"
    echo "  1. Request access to SAM3 at: https://huggingface.co/facebook/sam3"
    echo "  2. Get your token from: https://huggingface.co/settings/tokens"
    echo "  3. Update HF_TOKEN in .env file"
else
    print_success ".env file already exists"
fi

# Load HF_TOKEN from environment variable or .env file
# Priority: 1) Environment variable, 2) .env file
if [ -z "$HF_TOKEN" ]; then
    if [ -f ".env" ]; then
        print_info "Loading HF_TOKEN from .env file..."
        set -a
        source .env 2>/dev/null || true
        set +a
    fi
else
    print_info "Using HF_TOKEN from environment variable"
fi

# Verify Python version in venv
print_info "Verifying Python version in virtual environment..."
VENV_PYTHON_VERSION=$(python3 --version 2>&1 | sed -E 's/^Python ([0-9]+\.[0-9]+).*/\1/' | head -1)
print_info "Virtual environment Python version: $VENV_PYTHON_VERSION"

# Install all dependencies using uv sync
# uv sync will use the activated virtual environment's Python
# --no-install-project prevents trying to install this project itself (it's just a setup project)
print_info "Installing all dependencies using uv sync..."
print_info "This will install ultralytics, sam3, and all their dependencies from pyproject.toml"
uv sync --no-install-project
print_success "All dependencies installed via uv sync"

# Create models directory if it doesn't exist
print_info "Creating models directory..."
mkdir -p models
print_success "Models directory ready"

# Download SAM3 model weights
MODEL_PATH="${SAM3_MODEL_PATH:-models/sam3.pt}"
print_info "Downloading SAM3 model weights to $MODEL_PATH..."
if [ -f "$MODEL_PATH" ]; then
    print_warning "$MODEL_PATH already exists, skipping download"
else
    if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
        print_warning "HF_TOKEN not set in .env file. Skipping automatic download."
        print_warning "To download sam3.pt manually:"
        echo "  1. Set HF_TOKEN in .env file"
        echo "  2. Run: python3 -c \"from huggingface_hub import hf_hub_download; import os; hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', token=os.environ.get('HF_TOKEN'), local_dir='models')\""
    else
        print_info "Downloading sam3.pt from Hugging Face..."
        set +e  # Temporarily disable exit on error
        python3 << PYTHON_EOF
import os
import sys
from huggingface_hub import hf_hub_download

try:
    token = os.environ.get('HF_TOKEN', '')
    if not token or token == 'your_huggingface_token_here':
        print("HF_TOKEN not set or invalid")
        sys.exit(1)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Downloading sam3.pt...")
    checkpoint_path = hf_hub_download(
        repo_id="facebook/sam3",
        filename="sam3.pt",
        token=token,
        local_dir=models_dir
    )
    print(f"Downloaded to: {checkpoint_path}")
    sys.exit(0)
except Exception as e:
    print(f"Error downloading sam3.pt: {e}")
    print("This may be due to:")
    print("  - Invalid or missing HF_TOKEN")
    print("  - No access to facebook/sam3 repository (request access at https://huggingface.co/facebook/sam3)")
    print("  - Network issues")
    sys.exit(1)
PYTHON_EOF
        DOWNLOAD_EXIT_CODE=$?
        set -e  # Re-enable exit on error
        if [ $DOWNLOAD_EXIT_CODE -eq 0 ]; then
            print_success "SAM3 model weights downloaded successfully to $MODEL_PATH!"
        else
            print_warning "Failed to download sam3.pt automatically. Please download manually."
        fi
    fi
fi

# Copy BPE vocabulary file to models directory for easier access
print_info "Setting up BPE vocabulary file..."
BPE_SOURCE="submodules/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
BPE_TARGET="${SAM3_BPE_PATH:-models/bpe_simple_vocab_16e6.txt.gz}"

if [ -f "$BPE_TARGET" ]; then
    print_warning "BPE file already exists at $BPE_TARGET, skipping copy"
elif [ -f "$BPE_SOURCE" ]; then
    # Extract directory from target path
    BPE_DIR=$(dirname "$BPE_TARGET")
    mkdir -p "$BPE_DIR"
    cp "$BPE_SOURCE" "$BPE_TARGET"
    print_success "BPE vocabulary file copied to $BPE_TARGET"
else
    print_warning "BPE source file not found at $BPE_SOURCE"
    print_warning "The BPE file should be included in the sam3 package, but you can also download it from:"
    echo "  https://github.com/facebookresearch/sam3/blob/main/assets/bpe_simple_vocab_16e6.txt.gz"
fi

# Verify GPU availability with PyTorch (quick check)
print_info "Verifying GPU availability with PyTorch..."
set +e  # Temporarily disable exit on error for faster failure
if [ "$IS_MACOS" = true ]; then
    python3 << 'VERIFY_EOF'
import torch
import sys
import os

# Suppress verbose output for faster execution
os.environ['TORCH_LOGS'] = '+error'

try:
    # Check for MPS (Metal Performance Shaders) on macOS
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    if mps_available:
        print(f"PyTorch {torch.__version__}: MPS (Metal) available")
        print("✓ GPU support working (MPS)")
        sys.exit(0)
    else:
        print("✗ MPS not available")
        print("  PyTorch will use CPU (slower)")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
VERIFY_EOF
else
    python3 << 'VERIFY_EOF'
import torch
import sys
import os

# Suppress verbose output for faster execution
os.environ['TORCH_LOGS'] = '+error'

try:
    # Quick check - just verify CUDA is available without detailed info
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        # Only get detailed info if CUDA is available (faster path)
        print(f"PyTorch {torch.__version__}: CUDA available")
        print(f"CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
        gpu_count = torch.cuda.device_count()
        print(f"GPUs: {gpu_count}")
        # Only list first GPU to save time
        if gpu_count > 0:
            print(f"  GPU 0: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
            if gpu_count > 1:
                print(f"  ... and {gpu_count - 1} more GPU(s)")
        print("✓ GPU support working")
        sys.exit(0)
    else:
        print("✗ CUDA not available")
        print("  Check NVIDIA drivers and CUDA installation")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
VERIFY_EOF
fi
VERIFY_EXIT_CODE=$?
set -e  # Re-enable exit on error

if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    print_success "GPU verification passed!"
else
    if [ "$IS_MACOS" = true ]; then
        print_warning "MPS verification failed or not available."
        print_warning "Installation will continue, but GPU features may not work."
        print_warning "You can verify MPS support later with: python3 -c 'import torch; print(torch.backends.mps.is_available())'"
    else
        print_warning "GPU verification failed or CUDA not available."
        print_warning "Installation will continue, but GPU features may not work."
        print_warning "You can verify GPU support later with: python3 -c 'import torch; print(torch.cuda.is_available())'"
    fi
fi

print_success "Installation completed successfully!"
print_info "Summary:"
echo "  ✓ Virtual environment: .venv"
if [ "$IS_MACOS" = true ]; then
    echo "  ✓ PyTorch with MPS (Metal) support installed"
else
    echo "  ✓ PyTorch with CUDA support installed"
fi
echo "  ✓ All dependencies installed via uv sync (ultralytics, sam3, and dependencies)"
if [ -f "models/sam3.pt" ]; then
    echo "  ✓ SAM3 model weights (models/sam3.pt) downloaded"
else
    echo "  ⚠ SAM3 model weights (models/sam3.pt) not downloaded - check .env for HF_TOKEN"
fi
if [ -f "models/bpe_simple_vocab_16e6.txt.gz" ]; then
    echo "  ✓ BPE vocabulary file (models/bpe_simple_vocab_16e6.txt.gz) available"
else
    echo "  ⚠ BPE vocabulary file not found in models/ (should be available from package)"
fi
echo ""
print_info "To activate the virtual environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
if [ ! -f "models/sam3.pt" ]; then
    print_warning "Next steps:"
    echo "  1. Edit .env and add your Hugging Face token (HF_TOKEN)"
    echo "  2. Request access to SAM3 at: https://huggingface.co/facebook/sam3"
    echo "  3. Run the download command or restart this script"
fi

