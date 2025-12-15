# ultralytics-sam3-install

Installation setup for Ultralytics YOLO with SAM3 integration and GPU support.

## Overview

This repository provides a complete setup for installing and using Ultralytics YOLO with SAM3 (Segment Anything Model 3) integration. The installation is configured to utilize GPU acceleration, specifically optimized for systems with NVIDIA L40 GPUs.

## System Requirements

- **Python**: >= 3.9, < 3.13 (Python 3.12 recommended, automatically installed by uv)
- **NVIDIA GPU**: NVIDIA L40 or compatible (2 GPUs available on this server)
- **NVIDIA Drivers**: Latest drivers installed
- **CUDA Toolkit**: CUDA 11.8 or higher recommended
- **uv**: Python package installer (will be checked during installation)
- **Hugging Face Account**: Required for downloading SAM3 model weights (request access at https://huggingface.co/facebook/sam3)

## Project Structure

```
ultralytics-sam3-install/
├── submodules/
│   ├── ultralytics/      # Ultralytics YOLO repository
│   ├── sam3/             # Facebook Research SAM3 repository
│   ├── inference/        # Ultralytics inference repository
│   └── supervision/      # Roboflow supervision repository
├── models/               # Model files (created during installation)
│   ├── sam3.pt           # SAM3 model weights (downloaded automatically)
│   └── bpe_simple_vocab_16e6.txt.gz  # BPE vocabulary (copied automatically)
├── .venv/                # Virtual environment (created during installation)
├── .env                  # Configuration file (created during installation)
├── .env.example          # Example configuration file
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

### 2. Set Up Hugging Face Token

Before running the installation script, you need to:

1. **Request access** to the SAM3 model at: https://huggingface.co/facebook/sam3
2. **Get your Hugging Face token** from: https://huggingface.co/settings/tokens

The installation script will create a `.env` file template. After installation, edit `.env` and add your token:

```bash
HF_TOKEN=your_huggingface_token_here
```

Alternatively, you can copy the example file and edit it before installation:

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

**For Google Colab users**: The installation script automatically detects Colab and will try to load `HF_TOKEN` from Colab's userdata secrets. To set it up:

1. In Colab, go to the left sidebar → 🔑 **Secrets** (or use `from google.colab import userdata`)
2. Add a new secret named `HF_TOKEN` with your Hugging Face token as the value
3. The installation script will automatically use this token (priority: Colab userdata → Environment variable → `.env` file)

### 3. Run the Installation Script

The installation script will:
- Check for `uv` installation
- Initialize git submodules if needed
- Detect CUDA version and GPU availability
- Install Python 3.12 using `uv` (if not already available)
- Create a virtual environment using `uv venv` with Python 3.12
- Install PyTorch with CUDA support (skips if already installed to save time)
- Install all dependencies using `uv sync` (ultralytics, sam3, and all dependencies)
- Create `.env` file template if it doesn't exist
- Download SAM3 model weights (`sam3.pt`) to `models/` folder (if HF_TOKEN is set)
- Copy BPE vocabulary file to `models/` folder
- Verify GPU availability (quick check)

```bash
./install.sh
```

**Installation Time**: 
- First run: ~10-15 minutes (downloads PyTorch ~2-3 GB and all dependencies)
- Subsequent runs: ~2-5 minutes (skips PyTorch if already installed)

**Note**: If you haven't set your `HF_TOKEN` in `.env` before running the script, the model weights won't be downloaded automatically. You can:
1. Edit `.env` and add your token
2. Re-run the installation script, or
3. Manually download the model (see Troubleshooting section)

### 4. Activate the Virtual Environment

After installation, activate the virtual environment:

```bash
source .venv/bin/activate
```

## GPU Verification

The installation script automatically performs a quick GPU verification check. For a more detailed verification, you can manually run:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
```

## Usage

!!! success "Automatic Model Download"

    If you've set your `HF_TOKEN` in `.env` before running the installation script, the SAM3 model weights (`sam3.pt`) and BPE vocabulary file are automatically downloaded/copied to the `models/` folder during installation.

!!! note "Model File Locations"

    By default, the installation script places files in:
    - Model weights: `models/sam3.pt`
    - BPE vocabulary: `models/bpe_simple_vocab_16e6.txt.gz`
    
    You can customize these paths in the `.env` file using `SAM3_MODEL_PATH` and `SAM3_BPE_PATH`.

### Using SAM3

SAM 3 supports both Promptable Concept Segmentation (PCS) and Promptable Visual Segmentation (PVS) tasks. Once installed and activated, you can use SAM3 in Python:

#### Text-Based Concept Segmentation

Segment all instances of a concept using text descriptions:

```python
from ultralytics.models.sam.predict import SAM3SemanticPredictor

# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="models/sam3.pt",  # Path to downloaded model
    half=True,  # Use FP16 for faster inference
)
predictor = SAM3SemanticPredictor(
    overrides=overrides,
    bpe_path="models/bpe_simple_vocab_16e6.txt.gz",  # Path to BPE vocabulary
)

# Set image once for multiple queries
predictor.set_image("path/to/image.jpg")

# Query with text prompts to find all instances
results = predictor(text=["person", "bus", "glasses"], save=True)

# Works with descriptive phrases
results = predictor(text=["person with red cloth", "person with blue cloth"], save=True)
```

#### Visual Prompts (Points and Boxes)

Use point or box prompts for interactive segmentation (SAM 2-style):

```python
from ultralytics.models.sam.predict import SAM3Predictor

# Initialize predictor
overrides = dict(conf=0.25, task="segment", mode="predict", model="models/sam3.pt", half=True)
predictor = SAM3Predictor(overrides=overrides)

# Set image
predictor.set_image("path/to/image.jpg")

# Segment with point prompts
results = predictor(points=[[500, 375]], point_labels=[1], save=True)

# Segment with bounding box prompts
results = predictor(bboxes=[[100, 100, 200, 200]], save=True)

# Combine points and boxes
results = predictor(
    points=[[500, 375]],
    point_labels=[1],
    bboxes=[[100, 100, 200, 200]],
    save=True
)
```

#### Image Exemplar-Based Segmentation

Use bounding boxes as visual prompts to find all similar instances:

```python
from ultralytics.models.sam.predict import SAM3SemanticPredictor

# Initialize predictor
overrides = dict(conf=0.25, task="segment", mode="predict", model="models/sam3.pt", half=True)
predictor = SAM3SemanticPredictor(overrides=overrides, bpe_path="models/bpe_simple_vocab_16e6.txt.gz")

# Set image
predictor.set_image("path/to/image.jpg")

# Provide bounding box examples to segment similar objects
results = predictor(bboxes=[[480.0, 290.0, 590.0, 650.0]], save=True)
```

#### Video Concept Tracking

Track object instances across video frames:

```python
from ultralytics.models.sam.predict import SAM3VideoPredictor

# Create video predictor
overrides = dict(conf=0.25, task="segment", mode="predict", model="models/sam3.pt", half=True)
predictor = SAM3VideoPredictor(overrides=overrides)

# Track objects using bounding box prompts
results = predictor(
    source="path/to/video.mp4",
    bboxes=[[706.5, 442.5, 905.25, 555], [598, 635, 725, 750]],
    stream=True
)

# Process and display results
for r in results:
    r.show()  # Display frame with segmentation masks
```

### GPU Usage

SAM3 will automatically use available GPUs when running inference. To specify a GPU:

```python
from ultralytics.models.sam.predict import SAM3SemanticPredictor

# Initialize predictor with device specification
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="models/sam3.pt",
    device=0,  # Use GPU 0
    half=True,
)
predictor = SAM3SemanticPredictor(
    overrides=overrides,
    bpe_path="models/bpe_simple_vocab_16e6.txt.gz",
)

# Run inference on GPU
predictor.set_image("path/to/image.jpg")
results = predictor(text=["person"], save=True)
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

This project uses `uv` for Python package management. The `pyproject.toml` file defines project dependencies including:

- **Ultralytics** and **SAM3** (installed from submodules in editable mode)
- All SAM3 dependencies (timm, numpy, huggingface_hub, etc.)

To sync dependencies after making changes:

```bash
source .venv/bin/activate
uv sync --no-install-project
```

The `--no-install-project` flag prevents trying to install this project itself (it's just a setup project, not a Python package).

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
- **Python version issues**: The script automatically installs Python 3.12 using `uv python install`. If you encounter Python version errors, ensure `uv` has permission to install Python versions.
- **PyTorch installation slow**: PyTorch packages are large (~2-3 GB). The script shows progress during download. If PyTorch is already installed, the script will skip reinstallation to save time.
- **GPU verification warning**: If you see a GPU verification warning, installation will continue. You can verify GPU support later. This is usually not a critical issue unless you specifically need GPU features.
- **Model not downloaded**: If `sam3.pt` wasn't downloaded automatically:
  1. Ensure you've requested access to https://huggingface.co/facebook/sam3
  2. Check that `HF_TOKEN` is set correctly in `.env`
  3. Manually download using:
     ```bash
     source .venv/bin/activate
     python3 -c "from huggingface_hub import hf_hub_download; import os; hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', token=os.environ.get('HF_TOKEN'), local_dir='models')"
     ```

## License

This project is licensed under the MIT License. Note that Ultralytics is licensed under AGPL-3.0.

## References

- [Ultralytics Documentation](https://docs.ultralytics.com)
- [SAM3 Repository](https://github.com/facebookresearch/sam3)
- [uv Documentation](https://github.com/astral-sh/uv)
