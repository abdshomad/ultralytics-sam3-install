#!/usr/bin/env python3
"""
Test 04: Video Concept Tracking
Tests SAM3VideoPredictor with box prompts for object tracking across video frames.

This script implements the README example (lines 223-244) for video tracking,
displaying multiple frames side-by-side showing tracking consistency.
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

from ultralytics.models.sam.predict import SAM3VideoPredictor


def find_test_video():
    """Find a suitable test video from submodules."""
    possible_paths = [
        project_root / "submodules" / "sam3" / "assets" / "videos",
        project_root / "submodules" / "inference" / "assets",
    ]
    
    # Look for common video extensions
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    
    for base_path in possible_paths:
        if base_path.exists():
            for ext in video_extensions:
                videos = list(base_path.glob(f"*{ext}"))
                if videos:
                    return str(videos[0])
    
    return None  # Video is optional for this test


def check_requirements():
    """Check if required files exist."""
    model_path = project_root / "models" / "sam3.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please download sam3.pt to models/ directory"
        )
    
    return str(model_path)


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


def add_bbox_labels_to_frame(frame, bboxes):
    """Add coordinate labels to bounding boxes in frame."""
    frame_labeled = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
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
            label_y = cy - 5 if cy < frame_labeled.shape[0] // 2 else cy + text_height + 5
            
            # Adjust if label goes outside image bounds
            label_x = max(0, min(label_x, frame_labeled.shape[1] - text_width))
            label_y = max(text_height, min(label_y, frame_labeled.shape[0] - baseline))
            
            # Draw background rectangle
            cv2.rectangle(frame_labeled, 
                         (label_x - 2, label_y - text_height - 2), 
                         (label_x + text_width + 2, label_y + baseline + 2), 
                         (0, 0, 0), -1)
            
            # Draw coordinate text
            cv2.putText(frame_labeled, coord_text, (label_x, label_y), 
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return frame_labeled


def visualize_frames_side_by_side(frames_list, titles, output_path, bboxes=None):
    """Create side-by-side visualization of video frames."""
    num_frames = len(frames_list)
    if num_frames == 0:
        print("  ⚠ No frames to visualize")
        return
    
    fig, axes = plt.subplots(1, num_frames, figsize=(6 * num_frames, 6))
    if num_frames == 1:
        axes = [axes]
    
    for idx, (frame, title) in enumerate(zip(frames_list, titles)):
        # Add bbox labels if provided
        if bboxes is not None:
            frame = add_bbox_labels_to_frame(frame, bboxes)
        axes[idx].imshow(frame)
        axes[idx].set_title(title, fontsize=14, fontweight="bold")
        add_rulers_to_axis(axes[idx], frame.shape)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    """Main test function."""
    print("=" * 80)
    print("Test 04: Video Concept Tracking - SAM3VideoPredictor")
    print("=" * 80)
    
    # Check requirements
    print("\n[1/6] Checking requirements...")
    model_path = check_requirements()
    test_video = find_test_video()
    print(f"  ✓ Model: {model_path}")
    
    if test_video:
        print(f"  ✓ Test video: {test_video}")
    else:
        print("  ⚠ Test video not found - test will be skipped")
        print("  Note: Video tracking requires a test video file")
        print("  Place a video file in submodules/sam3/assets/videos/ or submodules/inference/assets/")
        return
    
    # Initialize predictor
    print("\n[2/6] Initializing predictor...")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
        half=True,
    )
    predictor = SAM3VideoPredictor(overrides=overrides)
    print("  ✓ Predictor initialized")
    
    # Define bounding box prompts (using example from README)
    bboxes = [[706.5, 442.5, 905.25, 555], [598, 635, 725, 750]]
    print(f"\n[3/6] Using bounding box prompts: {bboxes}")
    
    # Track objects across video
    print("\n[4/6] Running video tracking...")
    print("  Processing video frames (this may take a while)...")
    
    results = predictor(
        source=test_video,
        bboxes=bboxes,
        stream=True
    )
    
    # Collect frames for visualization (limit to first 5 frames)
    frames = []
    frame_titles = []
    max_frames = 5
    
    print("\n[5/6] Collecting frames for visualization...")
    for idx, r in enumerate(results):
        if idx >= max_frames:
            break
        
        # Get annotated frame
        annotated = r.plot()
        frames.append(annotated)
        frame_titles.append(f"Frame {idx + 1}")
        print(f"  ✓ Processed frame {idx + 1}")
    
    if len(frames) == 0:
        print("  ⚠ No frames were processed")
        return
    
    # Create visualizations
    print("\n[6/6] Creating side-by-side visualizations...")
    output_dir = project_root / "tests" / "v8.3.237" / "00-basic" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_frames_side_by_side(
        frames,
        frame_titles,
        output_dir / "04-video-tracking.png",
        bboxes=bboxes
    )
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
