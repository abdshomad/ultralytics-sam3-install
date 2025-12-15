#!/usr/bin/env python3
"""
Person Tracker with SAM3
Tracks persons across video frames using SAM3VideoSemanticPredictor,
displaying track IDs, trails, statistics, and confidence-based transparency.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# Add supervision submodule to path
supervision_path = project_root / "submodules" / "supervision"
if supervision_path.exists():
    sys.path.insert(0, str(supervision_path))

import supervision as sv
from supervision.annotators.utils import ColorLookup
from supervision.draw.color import ColorPalette

from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor


def draw_transparent_label(
    frame: np.ndarray,
    detections: sv.Detections,
    labels: list[str],
    color_lookup: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    """
    Draw labels with confidence-based transparent backgrounds.
    
    Args:
        frame: Input frame
        detections: Supervision detections
        labels: List of labels for each detection
        color_lookup: Dictionary mapping track_id to BGR color
        
    Returns:
        Annotated frame with transparent labels
    """
    annotated_frame = frame.copy()
    
    if len(detections) == 0:
        return annotated_frame
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    for idx, (xyxy, tracker_id, confidence) in enumerate(
        zip(detections.xyxy, detections.tracker_id, detections.confidence)
    ):
        if tracker_id is None:
            continue
            
        label = labels[idx] if idx < len(labels) else f"#{int(tracker_id)}"
        color = color_lookup.get(int(tracker_id), (255, 255, 255))
        
        # Calculate alpha based on confidence (lower confidence = more transparent)
        alpha = float(confidence)
        alpha = max(0.3, min(1.0, alpha))  # Clamp between 0.3 and 1.0
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Position label at top-left of bounding box
        x1, y1, x2, y2 = map(int, xyxy)
        label_x = x1
        label_y = y1 - 5 if y1 > 30 else y2 + text_height + 5
        
        # Ensure label stays within frame bounds
        label_y = max(text_height + 5, min(label_y, frame.shape[0] - baseline - 5))
        label_x = max(0, min(label_x, frame.shape[1] - text_width - 5))
        
        # Create background rectangle with transparency
        padding = 5
        bg_x1 = label_x - padding
        bg_y1 = label_y - text_height - padding
        bg_x2 = label_x + text_width + padding
        bg_y2 = label_y + baseline + padding
        
        # Ensure background stays within frame
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(frame.shape[1], bg_x2)
        bg_y2 = min(frame.shape[0], bg_y2)
        
        # Draw transparent background
        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1,
        )
        annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)
        
        # Draw text
        cv2.putText(
            annotated_frame,
            label,
            (label_x, label_y),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness,
            cv2.LINE_AA,
        )
    
    return annotated_frame


def draw_statistics_table(
    frame: np.ndarray,
    current_count: int,
    total_count: int,
    frame_time: float,
    total_time: float,
    avg_time: float,
) -> np.ndarray:
    """
    Draw statistics table with transparent background in top-left corner.
    
    Args:
        frame: Input frame
        current_count: Current persons in frame
        total_count: Total cumulative unique persons
        frame_time: Current frame inference time in ms
        total_time: Total inference time in seconds
        avg_time: Average inference time per frame in ms
        
    Returns:
        Frame with statistics table overlay
    """
    # Table rows
    rows = [
        f"Current Persons: {current_count}",
        f"Total Unique: {total_count}",
        f"Frame Time: {frame_time:.1f} ms",
        f"Total Time: {total_time:.2f} s",
        f"Avg Time: {avg_time:.1f} ms",
    ]
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 25
    padding = 10
    
    # Calculate table dimensions
    max_width = 0
    for row in rows:
        (text_width, _), _ = cv2.getTextSize(row, font, font_scale, thickness)
        max_width = max(max_width, text_width)
    
    table_width = max_width + 2 * padding
    table_height = len(rows) * line_height + 2 * padding
    
    # Create overlay for transparent background
    overlay = frame.copy()
    
    # Draw semi-transparent dark background
    cv2.rectangle(
        overlay,
        (padding, padding),
        (table_width, table_height),
        (0, 0, 0),
        -1,
    )
    
    # Blend with original frame (alpha ~0.7)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Draw text rows
    y_offset = padding + line_height
    for row in rows:
        cv2.putText(
            frame,
            row,
            (padding + 5, y_offset),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness,
            cv2.LINE_AA,
        )
        y_offset += line_height
    
    return frame


def generate_color_palette(num_colors: int = 50) -> list[tuple[int, int, int]]:
    """
    Generate a color palette with distinct colors.
    
    Args:
        num_colors: Number of colors to generate
        
    Returns:
        List of BGR color tuples
    """
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)
        # Convert HSV to BGR
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color_bgr)))
    return colors


def main():
    """Main function to run person tracking."""
    parser = argparse.ArgumentParser(
        description="Person Tracker with SAM3 - Track persons in video with trails and statistics"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/sam3.pt",
        help="Path to SAM3 model file (default: models/sam3.pt)",
    )
    parser.add_argument(
        "--bpe",
        type=str,
        default="models/bpe_simple_vocab_16e6.txt.gz",
        help="Path to BPE vocabulary file (default: models/bpe_simple_vocab_16e6.txt.gz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="01-person-tracker-with-sam3.mp4",
        help="Path to output video file (default: 01-person-tracker-with-sam3.mp4)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    bpe_path = Path(args.bpe)
    if not bpe_path.exists():
        print(f"Error: BPE vocabulary file not found: {bpe_path}")
        sys.exit(1)
    
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source video file not found: {source_path}")
        sys.exit(1)
    
    print("Initializing SAM3VideoSemanticPredictor...")
    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=str(model_path),
        half=True,
    )
    
    predictor = SAM3VideoSemanticPredictor(
        overrides=overrides,
        bpe_path=str(bpe_path),
    )
    print("Predictor initialized successfully")
    
    # Initialize tracking variables
    seen_track_ids = set()
    frame_inference_times = []
    total_inference_time = 0.0
    avg_time = 0.0  # Initialize to avoid UnboundLocalError when no frames processed
    start_time = time.time()
    
    # Generate color palette
    color_palette = generate_color_palette()
    color_lookup = {}
    
    # Initialize supervision annotators
    trace_annotator = sv.TraceAnnotator(
        trace_length=30,
        color_lookup=ColorLookup.TRACK,
    )
    mask_annotator = sv.MaskAnnotator(
        color=ColorPalette.DEFAULT,
    )
    
    # Get video info
    video_info = sv.VideoInfo.from_video_path(video_path=str(source_path))
    print(f"Video info: {video_info.width}x{video_info.height} @ {video_info.fps} fps, {video_info.total_frames} frames")
    
    # Process video
    print("Processing video...")
    results = predictor(source=str(source_path), text=["person"], stream=True)
    
    frame_count = 0
    
    with sv.VideoSink(target_path=args.output, video_info=video_info) as sink:
        for result in results:
            frame_count += 1
            
            # Record inference start time
            frame_start = time.time()
            
            # Convert Results to supervision Detections
            detections = sv.Detections.from_ultralytics(result)
            
            # Record inference end time
            frame_time = (time.time() - frame_start) * 1000  # Convert to ms
            frame_inference_times.append(frame_time)
            total_inference_time += frame_time
            
            # Calculate average
            avg_time = total_inference_time / len(frame_inference_times) if frame_inference_times else 0.0
            
            # Track unique IDs
            if detections.tracker_id is not None:
                seen_track_ids.update(detections.tracker_id.astype(int))
                
                # Assign colors to new track IDs
                for tid in detections.tracker_id.astype(int):
                    if tid not in color_lookup:
                        color_lookup[tid] = color_palette[tid % len(color_palette)]
            
            # Get original frame
            frame = result.orig_img.copy()
            
            # Create labels (only track ID)
            if detections.tracker_id is not None:
                labels = [f"#{int(tid)}" for tid in detections.tracker_id]
            else:
                labels = []
            
            # Annotate frame with trails
            annotated_frame = trace_annotator.annotate(
                scene=frame.copy(),
                detections=detections,
            )
            
            # Annotate frame with masks
            annotated_frame = mask_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
            )
            
            # Draw transparent labels
            annotated_frame = draw_transparent_label(
                frame=annotated_frame,
                detections=detections,
                labels=labels,
                color_lookup=color_lookup,
            )
            
            # Draw statistics table
            annotated_frame = draw_statistics_table(
                frame=annotated_frame,
                current_count=len(detections),
                total_count=len(seen_track_ids),
                frame_time=frame_time,
                total_time=total_inference_time / 1000.0,  # Convert to seconds
                avg_time=avg_time,
            )
            
            # Write frame
            sink.write_frame(frame=annotated_frame)
            
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames... (Avg: {avg_time:.1f} ms/frame)")
    
    # Calculate average if frames were processed
    if frame_count > 0:
        avg_time = total_inference_time / frame_count
    else:
        avg_time = 0.0
        print("\nWarning: No frames were processed. The video file may be corrupted or unreadable.")
    
    total_processing_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total frames: {frame_count}")
    if frame_count > 0:
        print(f"Total inference time: {total_inference_time / 1000.0:.2f} s")
        print(f"Average inference time: {avg_time:.1f} ms/frame")
    print(f"Total processing time: {total_processing_time:.2f} s")
    print(f"Total unique persons tracked: {len(seen_track_ids)}")
    if frame_count > 0:
        print(f"Output saved to: {args.output}")
    else:
        print(f"Warning: No output file created (no frames processed)")


if __name__ == "__main__":
    main()
