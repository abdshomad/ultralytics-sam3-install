#!/usr/bin/env python3
"""
Test 01: Text Prompts
Tests SAM3SemanticPredictor with text-based concept segmentation.

This script implements the README example (lines 150-174) for text-based segmentation,
displaying original image and segmentation results side-by-side.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import time

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


def add_time_text(img, elapsed_time):
    """Add elapsed time text to bottom right of image."""
    h, w = img.shape[:2]
    time_text = f"Inference: {elapsed_time:.3f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(time_text, font, font_scale, thickness)
    
    # Position at bottom right with padding
    x = w - text_width - 10
    y = h - 10
    
    # Draw background rectangle
    cv2.rectangle(img, (x - 5, y - text_height - 5), 
                 (x + text_width + 5, y + baseline + 5), (0, 0, 0), -1)
    
    # Draw text in white
    cv2.putText(img, time_text, (x, y), 
               font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img


def visualize_red_blue_cloth_combined(result_red, result_blue, original_img, elapsed_time=None):
    """Create combined visualization for both 'person with red cloth' and 'person with blue cloth'."""
    # Colors (RGB)
    red_color = (255, 0, 0)  # Pure red
    blue_color = (0, 0, 255)  # Pure blue
    red_label_color = (255, 255, 255)  # White (contrasting with red)
    blue_label_color = (255, 255, 0)  # Yellow (contrasting with blue)
    
    # Start with original image
    vis_img = original_img.copy()
    
    def draw_detections(result, color, label_color, label_text):
        """Helper function to draw detections with specified color."""
        nonlocal vis_img  # Allow modification of outer scope variable - must be first
        if result is None or result.masks is None or result.boxes is None:
            return
        
        masks = result.masks.data.cpu().numpy() if torch.is_tensor(result.masks.data) else result.masks.data
        boxes = result.boxes.xyxy.cpu().numpy() if torch.is_tensor(result.boxes.xyxy) else result.boxes.xyxy
        
        h, w = vis_img.shape[:2]
        
        # Process each detection
        for i in range(len(masks)):
            mask = masks[i]
            box = boxes[i]
            
            if len(mask.shape) == 2:
                # Resize mask to match image if needed
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = mask.astype(bool)
                else:
                    mask = mask.astype(bool)
                
                # Draw mask with specified color
                mask_overlay = vis_img.copy()
                mask_overlay[mask] = color
                vis_img = cv2.addWeighted(vis_img, 0.4, mask_overlay, 0.6, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = label_text
            if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
                conf = result.boxes.conf[i].item() if torch.is_tensor(result.boxes.conf[i]) else result.boxes.conf[i]
                label = f"{label} {conf:.2f}"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(vis_img, (x1, y1 - text_height - baseline - 5), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw label text in contrasting color
            cv2.putText(vis_img, label, (x1, y1 - baseline - 2), 
                       font, font_scale, label_color, thickness, cv2.LINE_AA)
    
    # Draw red cloth detections first
    if result_red is not None:
        draw_detections(result_red, red_color, red_label_color, "person with red cloth")
    
    # Draw blue cloth detections
    if result_blue is not None:
        draw_detections(result_blue, blue_color, blue_label_color, "person with blue cloth")
    
    # Add elapsed time to bottom right
    if elapsed_time is not None:
        vis_img = add_time_text(vis_img, elapsed_time)
    
    return vis_img


def visualize_side_by_side(original_img, results_list, titles, output_path, use_custom_blue=False, elapsed_time=None):
    """Create side-by-side visualization of original image and results."""
    num_results = len(results_list)
    fig, axes = plt.subplots(1, num_results + 1, figsize=(6 * (num_results + 1), 6))
    
    # Display original image
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    
    # Display each result
    for idx, (result, title) in enumerate(zip(results_list, titles), 1):
        if use_custom_blue and "red/blue cloth" in title.lower():
            # This should be handled separately with combined visualization
            # For now, use standard plot
            annotated = result.plot()
            if elapsed_time is not None:
                annotated = add_time_text(annotated, elapsed_time)
        else:
            # Get annotated image from result
            annotated = result.plot()
            # Add elapsed time to bottom right if provided
            if elapsed_time is not None:
                annotated = add_time_text(annotated, elapsed_time)
        axes[idx].imshow(annotated)
        axes[idx].set_title(title, fontsize=14, fontweight="bold")
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    """Main test function."""
    print("=" * 80)
    print("Test 01: Text Prompts - SAM3SemanticPredictor")
    print("=" * 80)
    
    # Check requirements
    print("\n[1/5] Checking requirements...")
    model_path, bpe_path = check_requirements()
    test_image = find_test_image()
    print(f"  ✓ Model: {model_path}")
    print(f"  ✓ BPE vocabulary: {bpe_path}")
    print(f"  ✓ Test image: {test_image}")
    
    # Initialize predictor
    print("\n[2/5] Initializing predictor...")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
        half=True,  # Use FP16 for faster inference
    )
    predictor = SAM3SemanticPredictor(
        overrides=overrides,
        bpe_path=bpe_path,
    )
    print("  ✓ Predictor initialized")
    
    # Test 1: Simple concepts - use bus.jpg image
    bus_image_path = project_root / "submodules" / "inference" / "assets" / "bus.jpg"
    if not bus_image_path.exists():
        print(f"  ⚠ Bus image not found at {bus_image_path}, using default test image")
        bus_image_path = test_image
    else:
        bus_image_path = str(bus_image_path)
    
    print("\n[3/6] Setting image for simple concepts test (bus.jpg)...")
    predictor.set_image(bus_image_path)
    original_img_simple = np.array(Image.open(bus_image_path))
    print(f"  ✓ Image set: {bus_image_path}")
    
    print("\n[4/6] Running text segmentation with simple concepts...")
    start_time = time.time()
    results_simple = predictor(text=["person", "bus", "glasses"], save=False)
    elapsed_time_simple = time.time() - start_time
    print(f"  ✓ Found {len(results_simple) if isinstance(results_simple, list) else 1} result(s)")
    print(f"  ✓ Inference time: {elapsed_time_simple:.3f}s")
    
    # Test 2: Descriptive phrases - use original test image
    print("\n[5/7] Resetting image for descriptive phrases test...")
    predictor.set_image(test_image)
    original_img_phrases = np.array(Image.open(test_image))
    print(f"  ✓ Image reset: {test_image}")
    
    # Run separate predictions for red and blue cloth
    print("\n[6/7] Running text segmentation for 'person with red cloth'...")
    start_time = time.time()
    results_red = predictor(text=["person with red cloth"], save=False)
    elapsed_time_red = time.time() - start_time
    print(f"  ✓ Found {len(results_red) if isinstance(results_red, list) else 1} result(s)")
    print(f"  ✓ Inference time: {elapsed_time_red:.3f}s")
    
    print("\n[7/7] Running text segmentation for 'person with blue cloth'...")
    predictor.set_image(test_image)  # Reset image
    start_time = time.time()
    results_blue = predictor(text=["person with blue cloth"], save=False)
    elapsed_time_blue = time.time() - start_time
    print(f"  ✓ Found {len(results_blue) if isinstance(results_blue, list) else 1} result(s)")
    print(f"  ✓ Inference time: {elapsed_time_blue:.3f}s")
    
    # Total elapsed time
    elapsed_time_phrases = elapsed_time_red + elapsed_time_blue
    
    # Create visualizations
    print("\n[8/8] Creating side-by-side visualizations...")
    output_dir = project_root / "tests" / "v8.3.237" / "00-basic" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle results - they might be single Results or list
    if not isinstance(results_simple, list):
        results_simple = [results_simple]
    if not isinstance(results_red, list):
        results_red = [results_red] if results_red is not None else []
    if not isinstance(results_blue, list):
        results_blue = [results_blue] if results_blue is not None else []
    
    # Visualize simple concepts (using bus.jpg image)
    visualize_side_by_side(
        original_img_simple,
        results_simple,
        ["Simple: person, bus, glasses"],
        output_dir / "01-text-prompts-simple.png",
        elapsed_time=elapsed_time_simple
    )
    
    # Visualize descriptive phrases with combined red and blue visualization
    result_red_single = results_red[0] if len(results_red) > 0 else None
    result_blue_single = results_blue[0] if len(results_blue) > 0 else None
    
    # Create combined visualization
    combined_img = visualize_red_blue_cloth_combined(
        result_red_single,
        result_blue_single,
        original_img_phrases,
        elapsed_time=elapsed_time_phrases
    )
    
    # Save combined visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img_phrases)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(combined_img)
    axes[1].set_title("Descriptive: person with red/blue cloth", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_dir / "01-text-prompts-phrases.png", dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_dir / '01-text-prompts-phrases.png'}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
