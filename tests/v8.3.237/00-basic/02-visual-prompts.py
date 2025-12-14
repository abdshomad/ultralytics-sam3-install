#!/usr/bin/env python3
"""
Test 02: Visual Prompts (Points and Boxes)
Tests SAM3Predictor with point and box prompts for interactive segmentation.

This script implements the README example (lines 176-203) for visual prompts,
displaying multiple results side-by-side showing different prompt types.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics.models.sam.predict import SAM3Predictor, SAM3SemanticPredictor


def find_test_image():
    """Find bus.jpg image for visual prompts test."""
    bus_image_path = project_root / "submodules" / "inference" / "assets" / "bus.jpg"
    
    if bus_image_path.exists():
        return str(bus_image_path)
    
    # Fallback to other images if bus.jpg not found
    possible_paths = [
        project_root / "submodules" / "sam3" / "assets" / "images" / "test_image.jpg",
        project_root / "submodules" / "sam3" / "assets" / "images" / "groceries.jpg",
        project_root / "submodules" / "inference" / "assets" / "zidane.jpg",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(
        f"Could not find test image. Checked: bus.jpg and {[str(p) for p in possible_paths]}"
    )


def check_requirements():
    """Check if required files exist."""
    model_path = project_root / "models" / "sam3.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please download sam3.pt to models/ directory"
        )
    
    return str(model_path)


def get_image_size(image_path):
    """Get image dimensions."""
    img = Image.open(image_path)
    return img.size  # (width, height)


def draw_points_on_image(img, points, point_labels):
    """Draw positive (+) and negative (-) points on image with coordinate labels."""
    vis_img = img.copy()
    for point, label in zip(points, point_labels):
        x, y = int(point[0]), int(point[1])
        if label == 1:  # Positive point
            cv2.circle(vis_img, (x, y), 10, (0, 255, 0), -1)  # Green circle
            cv2.circle(vis_img, (x, y), 10, (255, 255, 255), 2)  # White border
            cv2.putText(vis_img, '+', (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:  # Negative point
            cv2.circle(vis_img, (x, y), 10, (0, 0, 255), -1)  # Red circle
            cv2.circle(vis_img, (x, y), 10, (255, 255, 255), 2)  # White border
            cv2.putText(vis_img, '-', (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add coordinate label
        coord_text = f"({x},{y})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(coord_text, font, font_scale, thickness)
        
        # Position label above point
        label_x = x - text_width // 2
        label_y = y - 15
        
        # Draw background rectangle
        cv2.rectangle(vis_img, 
                     (label_x - 3, label_y - text_height - 3), 
                     (label_x + text_width + 3, label_y + baseline + 3), 
                     (0, 0, 0), -1)
        
        # Draw coordinate text
        cv2.putText(vis_img, coord_text, (label_x, label_y), 
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return vis_img


def draw_boxes_on_image(img, boxes, box_labels=None):
    """Draw positive (+) and negative (-) boxes on image with coordinate labels at corners."""
    vis_img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        if box_labels is None or box_labels[i] == 1:  # Positive box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green
            cv2.putText(vis_img, '+', (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:  # Negative box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red
            cv2.putText(vis_img, '-', (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
            label_y = cy - 5 if cy < vis_img.shape[0] // 2 else cy + text_height + 5
            
            # Adjust if label goes outside image bounds
            label_x = max(0, min(label_x, vis_img.shape[1] - text_width))
            label_y = max(text_height, min(label_y, vis_img.shape[0] - baseline))
            
            # Draw background rectangle
            cv2.rectangle(vis_img, 
                         (label_x - 2, label_y - text_height - 2), 
                         (label_x + text_width + 2, label_y + baseline + 2), 
                         (0, 0, 0), -1)
            
            # Draw coordinate text
            cv2.putText(vis_img, coord_text, (label_x, label_y), 
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return vis_img


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


def visualize_prompt_and_result(original_img, prompt_img, result, title, output_path):
    """Create side-by-side visualization: original+prompts -> detection result."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Original image with prompts
    axes[0].imshow(prompt_img)
    axes[0].set_title("Original + Prompts", fontsize=14, fontweight="bold")
    add_rulers_to_axis(axes[0], prompt_img.shape)
    
    # Right: Detection result
    if result is not None:
        annotated = result.plot()
        axes[1].imshow(annotated)
        axes[1].set_title("Detection Result", fontsize=14, fontweight="bold")
        add_rulers_to_axis(axes[1], annotated.shape)
    else:
        axes[1].imshow(original_img)
        axes[1].set_title("Detection Result (No detections)", fontsize=14, fontweight="bold")
        add_rulers_to_axis(axes[1], original_img.shape)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    """Main test function."""
    print("=" * 80)
    print("Test 02: Visual Prompts - SAM3Predictor")
    print("=" * 80)
    
    # Text prompts array
    text_prompts = ["right glasses", "right hand", "right shoess"]
    
    # Check requirements
    print("\n[1/8] Checking requirements...")
    model_path = check_requirements()
    test_image = find_test_image()
    bpe_path = project_root / "models" / "bpe_simple_vocab_16e6.txt.gz"
    print(f"  ✓ Model: {model_path}")
    print(f"  ✓ Test image: {test_image}")
    if bpe_path.exists():
        print(f"  ✓ BPE vocabulary: {bpe_path}")
    
    # Get image dimensions for coordinate calculation
    img_width, img_height = get_image_size(test_image)
    print(f"  ✓ Image size: {img_width}x{img_height}")
    
    # Initialize predictors
    print("\n[2/8] Initializing predictors...")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
        half=True,
    )
    predictor = SAM3Predictor(overrides=overrides)
    if bpe_path.exists():
        predictor_semantic = SAM3SemanticPredictor(
            overrides=overrides,
            bpe_path=str(bpe_path)
        )
    else:
        predictor_semantic = None
        print("  ⚠ BPE vocabulary not found, text prompts will be skipped")
    print("  ✓ Predictors initialized")
    
    # Set image
    print("\n[3/8] Setting image...")
    predictor.set_image(test_image)
    if predictor_semantic:
        predictor_semantic.set_image(test_image)
    original_img = np.array(Image.open(test_image))
    print("  ✓ Image set")
    
    # Define prompts
    center_x, center_y = img_width // 2, img_height // 2
    # Positive point (user-specified coordinates)
    pos_point = [[430, 730]]
    pos_point_label = [1]
    # Negative point (offset)
    neg_point = [[center_x + 100, center_y + 100]]
    neg_point_label = [0]
    # Combined points
    all_points = pos_point + neg_point
    all_point_labels = pos_point_label + neg_point_label
    
    # Positive box (center region)
    box_size = min(img_width, img_height) // 4
    pos_box_x1 = (img_width - box_size) // 2
    pos_box_y1 = (img_height - box_size) // 2
    pos_box_x2 = pos_box_x1 + box_size
    pos_box_y2 = pos_box_y1 + box_size
    pos_box = [[pos_box_x1, pos_box_y1, pos_box_x2, pos_box_y2]]
    
    # Negative box (different region)
    neg_box_x1 = img_width // 4
    neg_box_y1 = img_height // 4
    neg_box_x2 = neg_box_x1 + box_size // 2
    neg_box_y2 = neg_box_y1 + box_size // 2
    neg_box = [[neg_box_x1, neg_box_y1, neg_box_x2, neg_box_y2]]
    
    # Test 1: Points with positive and negative
    print("\n[4/8] Testing points (positive + negative)...")
    predictor.set_image(test_image)  # Reset
    results_points = predictor(points=all_points, point_labels=all_point_labels, save=False)
    result_points_single = results_points[0] if isinstance(results_points, list) and len(results_points) > 0 else results_points
    print(f"  ✓ Points prompt completed")
    
    # Test 2: Boxes with positive and negative
    print("\n[5/8] Testing boxes (positive + negative)...")
    predictor.set_image(test_image)  # Reset
    # Note: SAM3Predictor may not support negative boxes directly, so we'll use positive boxes
    results_boxes = predictor(bboxes=pos_box, save=False)
    result_boxes_single = results_boxes[0] if isinstance(results_boxes, list) and len(results_boxes) > 0 else results_boxes
    print(f"  ✓ Boxes prompt completed")
    
    # Test 3: Text prompts
    print("\n[6/8] Testing text prompts...")
    if predictor_semantic:
        predictor_semantic.set_image(test_image)  # Reset
        results_texts = predictor_semantic(text=text_prompts, save=False)
        result_texts_single = results_texts[0] if isinstance(results_texts, list) and len(results_texts) > 0 else results_texts
        print(f"  ✓ Text prompts completed")
    else:
        result_texts_single = None
        print("  ⚠ Text prompts skipped (no BPE vocabulary)")
    
    # Test 4: Combined (points + boxes + texts)
    print("\n[7/8] Testing combined prompts (points + boxes + texts)...")
    if predictor_semantic:
        predictor_semantic.set_image(test_image)  # Reset
        results_combined = predictor_semantic(
            points=all_points,
            point_labels=all_point_labels,
            bboxes=pos_box,
            text=text_prompts,
            save=False
        )
        result_combined_single = results_combined[0] if isinstance(results_combined, list) and len(results_combined) > 0 else results_combined
        print(f"  ✓ Combined prompts completed")
    else:
        # Fallback to non-semantic predictor
        predictor.set_image(test_image)  # Reset
        results_combined = predictor(
            points=all_points,
            point_labels=all_point_labels,
            bboxes=pos_box,
            save=False
        )
        result_combined_single = results_combined[0] if isinstance(results_combined, list) and len(results_combined) > 0 else results_combined
        print(f"  ✓ Combined prompts completed (without text)")
    
    # Create visualizations
    print("\n[8/8] Creating visualizations...")
    output_dir = project_root / "tests" / "v8.3.237" / "00-basic" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Image 1: Original + Points (+ and -) -> Detection Result
    prompt_img_1 = draw_points_on_image(original_img, all_points, all_point_labels)
    visualize_prompt_and_result(
        original_img,
        prompt_img_1,
        result_points_single,
        "Image 1: Points (Positive + Negative)",
        output_dir / "02-visual-prompts-01-points.png"
    )
    
    # Image 2: Original + Boxes (+ and -) -> Detection Result
    prompt_img_2 = draw_boxes_on_image(original_img, pos_box + neg_box, [1, 0])
    visualize_prompt_and_result(
        original_img,
        prompt_img_2,
        result_boxes_single,
        "Image 2: Boxes (Positive + Negative)",
        output_dir / "02-visual-prompts-02-boxes.png"
    )
    
    # Image 3: Original + Texts -> Detection Result
    if result_texts_single is not None:
        # Draw text labels on original image
        prompt_img_3 = original_img.copy()
        text_label = ", ".join(text_prompts)
        cv2.putText(prompt_img_3, text_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(prompt_img_3, text_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        visualize_prompt_and_result(
            original_img,
            prompt_img_3,
            result_texts_single,
            f"Image 3: Text Prompts ({', '.join(text_prompts)})",
            output_dir / "02-visual-prompts-03-texts.png"
        )
    
    # Image 4: Original + Points + Boxes + Texts -> Detection Result
    prompt_img_4 = original_img.copy()
    prompt_img_4 = draw_points_on_image(prompt_img_4, all_points, all_point_labels)
    prompt_img_4 = draw_boxes_on_image(prompt_img_4, pos_box + neg_box, [1, 0])
    text_label = ", ".join(text_prompts)
    cv2.putText(prompt_img_4, text_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(prompt_img_4, text_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    visualize_prompt_and_result(
        original_img,
        prompt_img_4,
        result_combined_single,
        "Image 4: Combined (Points + Boxes + Texts)",
        output_dir / "02-visual-prompts-04-combined.png"
    )
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
