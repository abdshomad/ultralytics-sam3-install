#!/usr/bin/env python3
"""
Test 05: GPU Usage
Tests SAM3SemanticPredictor with explicit GPU device specification.

This script implements the README example (lines 246-270) for GPU usage,
displaying results and GPU utilization information side-by-side.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics.models.sam.predict import SAM3SemanticPredictor


def find_test_image():
    """Find a suitable test image from submodules."""
    possible_paths = [
        project_root / "submodules" / "sam3" / "assets" / "images" / "test_image.jpg",
        project_root / "submodules" / "inference" / "assets" / "bus.jpg",
        project_root / "submodules" / "sam3" / "assets" / "images" / "groceries.jpg",
        project_root / "submodules" / "inference" / "assets" / "zidane.jpg",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(
        f"Could not find test image. Checked: {[str(p) for p in possible_paths]}"
    )


def check_requirements():
    """Check if required files exist."""
    model_path = project_root / "models" / "sam3.pt"
    bpe_path = project_root / "models" / "bpe_simple_vocab_16e6.txt.gz"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please download sam3.pt to models/ directory"
        )
    
    if not bpe_path.exists():
        raise FileNotFoundError(
            f"BPE vocabulary not found: {bpe_path}\n"
            "Please ensure bpe_simple_vocab_16e6.txt.gz is in models/ directory"
        )
    
    return str(model_path), str(bpe_path)


def check_gpu_availability():
    """Check GPU availability and return device info."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"  ✓ GPU available: {device_name}")
        print(f"  ✓ GPU count: {device_count}")
        return True, 0
    else:
        print("  ⚠ GPU not available, will use CPU")
        return False, "cpu"


def add_rulers_to_axis(ax, img_shape):
    """Add x and y axis rulers to matplotlib axis."""
    h, w = img_shape[:2]
    # Enable axes
    ax.axis("on")
    # Set limits
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Invert y-axis for image coordinates
    # Add labels
    ax.set_xlabel("X (pixels)", fontsize=10)
    ax.set_ylabel("Y (pixels)", fontsize=10)
    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    # Set tick spacing
    ax.set_xticks(range(0, w, max(50, w // 10)))
    ax.set_yticks(range(0, h, max(50, h // 10)))


def visualize_side_by_side(original_img, result, gpu_info, output_path):
    """Create side-by-side visualization of original image, results, and GPU info."""
    fig = plt.figure(figsize=(16, 8))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_img)
    ax1.set_title("Original Image", fontsize=14, fontweight="bold")
    add_rulers_to_axis(ax1, original_img.shape)
    
    # Results
    ax2 = fig.add_subplot(gs[0, 1])
    annotated = result.plot()
    ax2.imshow(annotated)
    ax2.set_title("Segmentation Results", fontsize=14, fontweight="bold")
    add_rulers_to_axis(ax2, annotated.shape)
    
    # GPU information
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    
    info_text = "GPU Information:\n"
    info_text += f"  CUDA Available: {torch.cuda.is_available()}\n"
    if torch.cuda.is_available():
        info_text += f"  Device Name: {torch.cuda.get_device_name(0)}\n"
        info_text += f"  Device Count: {torch.cuda.device_count()}\n"
        info_text += f"  Current Device: {torch.cuda.current_device()}\n"
        info_text += f"  Device Used: {gpu_info}\n"
        # Get memory info
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            info_text += f"  Memory Allocated: {memory_allocated:.2f} GB\n"
            info_text += f"  Memory Reserved: {memory_reserved:.2f} GB\n"
    else:
        info_text += "  Using CPU\n"
    
    ax3.text(0.1, 0.5, info_text, fontsize=12, family="monospace",
             verticalalignment="center", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.suptitle("GPU Usage Test Results", fontsize=16, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    """Main test function."""
    print("=" * 80)
    print("Test 05: GPU Usage - SAM3SemanticPredictor")
    print("=" * 80)
    
    # Check GPU availability
    print("\n[1/6] Checking GPU availability...")
    gpu_available, device = check_gpu_availability()
    
    # Check requirements
    print("\n[2/6] Checking requirements...")
    model_path, bpe_path = check_requirements()
    test_image = find_test_image()
    print(f"  ✓ Model: {model_path}")
    print(f"  ✓ BPE vocabulary: {bpe_path}")
    print(f"  ✓ Test image: {test_image}")
    
    # Initialize predictor with explicit device
    print("\n[3/6] Initializing predictor with device specification...")
    device_id = device if isinstance(device, int) else 0
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
        device=device_id,  # Use GPU 0 or CPU
        half=True,
    )
    predictor = SAM3SemanticPredictor(
        overrides=overrides,
        bpe_path=bpe_path,
    )
    print(f"  ✓ Predictor initialized with device: {device_id}")
    
    # Set image
    print("\n[4/6] Setting image...")
    predictor.set_image(test_image)
    original_img = np.array(Image.open(test_image))
    print("  ✓ Image set")
    
    # Run inference
    print("\n[5/6] Running inference on GPU...")
    results = predictor(text=["person"], save=False)
    print(f"  ✓ Inference completed")
    
    # Handle results - they might be single Results or list
    if isinstance(results, list):
        result = results[0] if len(results) > 0 else None
    else:
        result = results
    
    if result is None:
        print("  ⚠ Warning: No results returned")
        return
    
    # Create visualizations
    print("\n[6/6] Creating side-by-side visualizations...")
    output_dir = project_root / "tests" / "v8.3.237" / "00-basic" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_side_by_side(
        original_img,
        result,
        device_id,
        output_dir / "05-gpu-usage.png"
    )
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
