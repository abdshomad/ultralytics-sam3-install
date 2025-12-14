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


def visualize_frames_side_by_side(frames_list, titles, output_path):
    """Create side-by-side visualization of video frames."""
    num_frames = len(frames_list)
    if num_frames == 0:
        print("  ⚠ No frames to visualize")
        return
    
    fig, axes = plt.subplots(1, num_frames, figsize=(6 * num_frames, 6))
    if num_frames == 1:
        axes = [axes]
    
    for idx, (frame, title) in enumerate(zip(frames_list, titles)):
        axes[idx].imshow(frame)
        axes[idx].set_title(title, fontsize=14, fontweight="bold")
        axes[idx].axis("off")
    
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
        output_dir / "04-video-tracking.png"
    )
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
