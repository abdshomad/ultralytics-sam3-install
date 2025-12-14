# ultralytics-sam3-install

Installation setup for Ultralytics YOLO with SAM3 integration and GPU support.

## Overview

This repository provides a complete setup for installing and using Ultralytics YOLO with SAM3 (Segment Anything Model 3) integration. The installation is configured to utilize GPU acceleration, specifically optimized for systems with NVIDIA L40 GPUs.

## System Requirements

- **Python**: >= 3.8
- **NVIDIA GPU**: NVIDIA L40 or compatible (2 GPUs available on this server)
- **NVIDIA Drivers**: Latest drivers installed
- **CUDA Toolkit**: CUDA 11.8 or higher recommended
- **uv**: Python package installer (will be checked during installation)

## Project Structure

```
ultralytics-sam3-install/
├── submodules/
│   ├── ultralytics/      # Ultralytics YOLO repository
│   ├── sam3/             # Facebook Research SAM3 repository
│   ├── inference/        # Ultralytics inference repository
│   └── supervision/      # Roboflow supervision repository
├── install.sh            # Installation script
├── pyproject.toml        # Python dependency management
├── AGENTS.md            # Cursor IDE configuration
└── README.md            # This file
```

## Installation

### 1. Clone the Repository with Submodules

If you're cloning this repository for the first time:

```bash
git clone --recurse-submodules https://github.com/your-username/ultralytics-sam3-install.git
cd ultralytics-sam3-install
```

If you've already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Run the Installation Script

The installation script will:
- Check for `uv` installation
- Initialize git submodules if needed
- Detect CUDA version and GPU availability
- Create a virtual environment using `uv venv`
- Install PyTorch with CUDA support
- Install Ultralytics from the local submodule in editable mode
- Verify GPU availability

```bash
./install.sh
```

### 3. Activate the Virtual Environment

After installation, activate the virtual environment:

```bash
source .venv/bin/activate
```

## GPU Verification

The installation script automatically verifies GPU availability. You can also manually verify by running:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

## Usage

### Using Ultralytics YOLO

Once installed, you can use Ultralytics YOLO from the command line:

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run YOLO commands
yolo predict model=yolo11n.pt source=path/to/image.jpg
```

Or in Python:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolo11n.pt')

# Run inference
results = model('path/to/image.jpg')
```

### GPU Usage

Ultralytics will automatically use available GPUs when running inference or training. To specify a GPU:

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model('path/to/image.jpg', device=0)  # Use GPU 0
```

For multi-GPU training:

```python
model.train(data='coco.yaml', device=[0, 1])  # Use both GPUs
```

## Submodules

This project includes the following git submodules:

- **[ultralytics](https://github.com/ultralytics/ultralytics)**: Ultralytics YOLO framework
- **[sam3](https://github.com/facebookresearch/sam3)**: Segment Anything Model 3
- **[inference](https://github.com/ultralytics/inference)**: Ultralytics inference engine
- **[supervision](https://github.com/roboflow/supervision)**: Roboflow supervision utilities

### Updating Submodules

To update all submodules to their latest commits:

```bash
git submodule update --remote --recursive
```

## Development

This project uses `uv` for Python package management. The `pyproject.toml` file defines project dependencies and can be used with `uv sync`:

```bash
uv sync
```

## Troubleshooting

### GPU Not Detected

If the installation script reports that GPUs are not available:

1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```

2. Check CUDA installation:
   ```bash
   nvcc --version
   ```

3. Verify PyTorch CUDA support:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

### Installation Issues

- **uv not found**: Install uv using `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **CUDA version mismatch**: The script attempts to detect CUDA version automatically. If issues persist, you may need to manually install the correct PyTorch version.

## License

This project is licensed under the MIT License. Note that Ultralytics is licensed under AGPL-3.0.

## References

- [Ultralytics Documentation](https://docs.ultralytics.com)
- [SAM3 Repository](https://github.com/facebookresearch/sam3)
- [uv Documentation](https://github.com/astral-sh/uv)
