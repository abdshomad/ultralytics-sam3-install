#!/usr/bin/env python3
"""
Test 03: Image Exemplar-Based Segmentation
Tests SAM3SemanticPredictor with exemplar bounding boxes to find similar instances.

This script implements the README example (lines 205-221) for exemplar-based segmentation,
displaying original image with exemplar box and all detected similar instances side-by-side.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics.models.sam.predict import SAM3SemanticPredictor


def find_test_image():
    """Find a suitable test image with multiple similar objects."""
    possible_paths = [
        project_root / "submodules" / "sam3" / "assets" / "images" / "groceries.jpg",
        project_root / "submodules" / "sam3" / "assets" / "images" / "test_image.jpg",
        project_root / "submodules" / "inference" / "assets" / "bus.jpg",
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


def draw_exemplar_box(img_array, bbox, color=(255, 0, 0), width=3):
    """Draw exemplar bounding box on image with coordinate labels at corners."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = map(int, bbox)
    # Draw rectangle
    for i in range(width):
        draw.rectangle(
            [x1 - i, y1 - i, x2 + i, y2 + i],
            outline=color,
            width=1
        )
    
    # Convert to numpy for OpenCV text drawing
    img_np = np.array(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    # Add coordinate labels at all four corners
    corners = [
        ((x1, y1), f"({x1},{y1})"),  # Top-left
        ((x2, y1), f"({x2},{y1})"),  # Top-right
        ((x1, y2), f"({x1},{y2})"),  # Bottom-left
        ((x2, y2), f"({x2},{y2})"),  # Bottom-right
    ]
    
    for (cx, cy), coord_text in corners:
        (text_width, text_height), baseline = cv2.getTextSize(coord_text, font, font_scale, thickness)
        
        # Position label near corner
        label_x = cx - text_width // 2
        label_y = cy - 5 if cy < img_np.shape[0] // 2 else cy + text_height + 5
        
        # Adjust if label goes outside image bounds
        label_x = max(0, min(label_x, img_np.shape[1] - text_width))
        label_y = max(text_height, min(label_y, img_np.shape[0] - baseline))
        
        # Draw background rectangle
        cv2.rectangle(img_np, 
                     (label_x - 2, label_y - text_height - 2), 
                     (label_x + text_width + 2, label_y + baseline + 2), 
                     (0, 0, 0), -1)
        
        # Draw coordinate text
        cv2.putText(img_np, coord_text, (label_x, label_y), 
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img_np


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


def visualize_side_by_side(original_img, exemplar_img, result, output_path):
    """Create side-by-side visualization of original, exemplar, and results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display original image
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    add_rulers_to_axis(axes[0], original_img.shape)
    
    # Display exemplar box
    axes[1].imshow(exemplar_img)
    axes[1].set_title("Exemplar Box", fontsize=14, fontweight="bold")
    add_rulers_to_axis(axes[1], exemplar_img.shape)
    
    # Display result
    annotated = result.plot()
    axes[2].imshow(annotated)
    axes[2].set_title("All Similar Instances", fontsize=14, fontweight="bold")
    add_rulers_to_axis(axes[2], annotated.shape)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    """Main test function."""
    print("=" * 80)
    print("Test 03: Image Exemplar-Based Segmentation - SAM3SemanticPredictor")
    print("=" * 80)
    
    # Check requirements
    print("\n[1/6] Checking requirements...")
    model_path, bpe_path = check_requirements()
    test_image = find_test_image()
    print(f"  ✓ Model: {model_path}")
    print(f"  ✓ BPE vocabulary: {bpe_path}")
    print(f"  ✓ Test image: {test_image}")
    
    # Get image dimensions for exemplar box
    img = Image.open(test_image)
    img_width, img_height = img.size
    print(f"  ✓ Image size: {img_width}x{img_height}")
    
    # Initialize predictor
    print("\n[2/6] Initializing predictor...")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
        half=True,
    )
    predictor = SAM3SemanticPredictor(
        overrides=overrides,
        bpe_path=bpe_path,
    )
    print("  ✓ Predictor initialized")
    
    # Set image
    print("\n[3/6] Setting image...")
    predictor.set_image(test_image)
    original_img = np.array(img)
    print("  ✓ Image set")
    
    # Define exemplar box (using example from README or center region)
    # Using example coordinates from README if image is large enough, otherwise center region
    if img_width > 600 and img_height > 700:
        exemplar_bbox = [480.0, 290.0, 590.0, 650.0]
    else:
        # Use center region as exemplar
        box_size_w = img_width // 4
        box_size_h = img_height // 4
        exemplar_bbox = [
            (img_width - box_size_w) // 2,
            (img_height - box_size_h) // 2,
            (img_width + box_size_w) // 2,
            (img_height + box_size_h) // 2,
        ]
    
    print(f"\n[4/6] Using exemplar box: {exemplar_bbox}")
    
    # Draw exemplar box on image
    exemplar_img = draw_exemplar_box(original_img.copy(), exemplar_bbox)
    
    # Test exemplar-based segmentation
    print("\n[5/6] Running exemplar-based segmentation...")
    results = predictor(bboxes=[exemplar_bbox], save=False)
    print(f"  ✓ Segmentation completed")
    
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
        exemplar_img,
        result,
        output_dir / "03-exemplar-segmentation.png"
    )
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
