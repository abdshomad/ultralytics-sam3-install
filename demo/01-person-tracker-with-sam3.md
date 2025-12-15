# Person Tracker with SAM3 - Usage Guide

This script tracks persons across video frames using SAM3 (Segment Anything Model 3), displaying track IDs, trails, statistics, and confidence-based transparency. The script automatically processes video in three predefined time segments.

## Features

- **Automatic YouTube Video Download**: Downloads videos from YouTube URLs automatically
- **Multi-Segment Processing**: Processes video in 3 predefined time segments:
  - Segment 1: 00:26 - 00:42 (16 seconds)
  - Segment 2: 01:00 - 01:24 (24 seconds)
  - Segment 3: 01:48 - 02:32 (44 seconds)
- **Person Tracking**: Tracks persons across frames with unique IDs
- **Visual Annotations**: 
  - Color-coded tracking trails
  - Segmentation masks
  - Track ID labels with confidence-based transparency
  - Real-time statistics overlay

## Prerequisites

### Required Software

1. **Python Environment**: Python 3.9-3.12 with `uv` package manager
2. **ffmpeg**: For video segment extraction
   ```bash
   sudo apt install ffmpeg
   ```
3. **yt-dlp**: For YouTube video downloads (installed via `uv`)
   ```bash
   uv pip install yt-dlp
   ```

### Required Files

- **SAM3 Model**: `models/sam3.pt`
- **BPE Vocabulary**: `models/bpe_simple_vocab_16e6.txt.gz`

These should be in the project root `models/` directory.

## Usage

### Basic Usage (Default YouTube Video)

The simplest way to use the script is without any arguments. It will use the default YouTube video:

```bash
./demo/01-person-tracker-with-sam3.sh
```

This will:
1. Download the default YouTube video as `01-person-tracker-with-sam3-original-video.mp4` (if not already downloaded)
2. Extract 3 time segments using ffmpeg
3. Process each segment and generate detection result videos
4. All files follow the naming convention: `01-person-tracker-with-sam3-original-video-{time-range}[-detection-result].mp4`

### Using a Custom YouTube URL

```bash
./demo/01-person-tracker-with-sam3.sh --source "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

### Using a Local Video File

```bash
./demo/01-person-tracker-with-sam3.sh --source "/path/to/your/video.mp4"
```

### Custom Output Directory

```bash
./demo/01-person-tracker-with-sam3.sh --output "output/my_tracking_results.mp4"
```

This will generate files in the `output/` directory:
- `01-person-tracker-with-sam3-original-video-00-26-to-00-42.mp4` (segment file)
- `01-person-tracker-with-sam3-original-video-00-26-to-00-42-detection-result.mp4` (result)
- `01-person-tracker-with-sam3-original-video-01-00-to-01-24.mp4` (segment file)
- `01-person-tracker-with-sam3-original-video-01-00-to-01-24-detection-result.mp4` (result)
- `01-person-tracker-with-sam3-original-video-01-48-to-02-32.mp4` (segment file)
- `01-person-tracker-with-sam3-original-video-01-48-to-02-32-detection-result.mp4` (result)

### Custom Model and BPE Path

```bash
./demo/01-person-tracker-with-sam3.sh \
    --model "path/to/custom/sam3.pt" \
    --bpe "path/to/custom/bpe_simple_vocab_16e6.txt.gz"
```

### Adjust Confidence Threshold

```bash
./demo/01-person-tracker-with-sam3.sh --conf 0.5
```

Higher values (0.0-1.0) mean stricter detection requirements.

### Complete Example

```bash
./demo/01-person-tracker-with-sam3.sh \
    --source "https://www.youtube.com/watch?v=qjSUk9o-D6E&t=1s" \
    --output "results/person_tracking.mp4" \
    --model "models/sam3.pt" \
    --bpe "models/bpe_simple_vocab_16e6.txt.gz" \
    --conf 0.25
```

This will:
1. Download the video as `01-person-tracker-with-sam3-original-video.mp4` (if not already downloaded)
2. Extract segments and save them in the `results/` directory
3. Process each segment and save detection results in the `results/` directory
4. Generate files following the naming convention: `01-person-tracker-with-sam3-original-video-{time-range}[-detection-result].mp4`

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--source` | No* | Default YouTube URL | Path to input video file or YouTube URL |
| `--output` | No | Current directory | Output directory path (files will be named automatically) |
| `--model` | No | `models/sam3.pt` | Path to SAM3 model file |
| `--bpe` | No | `models/bpe_simple_vocab_16e6.txt.gz` | Path to BPE vocabulary file |
| `--conf` | No | `0.25` | Confidence threshold (0.0-1.0) |

\* If `--source` is not specified, the default YouTube URL will be used.

## Output Files

The script uses a consistent naming convention based on the script name. All files are named as `{script-name}-original-video-{time-range}[-detection-result].mp4`.

### File Naming Convention

- **Downloaded original video**: `{script-name}-original-video.mp4`
  - Example: `01-person-tracker-with-sam3-original-video.mp4`

- **Segment files**: `{script-name}-original-video-{from-time}-to-{end-time}.mp4`
  - Example: `01-person-tracker-with-sam3-original-video-00-26-to-00-42.mp4`

- **Detection result files**: `{script-name}-original-video-{from-time}-to-{end-time}-detection-result.mp4`
  - Example: `01-person-tracker-with-sam3-original-video-00-26-to-00-42-detection-result.mp4`

### Generated Files for Each Segment

1. **Segment 1** (00:26 - 00:42):
   - `01-person-tracker-with-sam3-original-video-00-26-to-00-42.mp4` (extracted segment)
   - `01-person-tracker-with-sam3-original-video-00-26-to-00-42-detection-result.mp4` (tracking result)

2. **Segment 2** (01:00 - 01:24):
   - `01-person-tracker-with-sam3-original-video-01-00-to-01-24.mp4` (extracted segment)
   - `01-person-tracker-with-sam3-original-video-01-00-to-01-24-detection-result.mp4` (tracking result)

3. **Segment 3** (01:48 - 02:32):
   - `01-person-tracker-with-sam3-original-video-01-48-to-02-32.mp4` (extracted segment)
   - `01-person-tracker-with-sam3-original-video-01-48-to-02-32-detection-result.mp4` (tracking result)

Each detection result video contains:
- Person segmentation masks
- Tracking trails showing movement paths
- Track ID labels (#1, #2, etc.) with confidence-based transparency
- Statistics overlay showing:
  - Current persons in frame
  - Total unique persons tracked
  - Frame processing time
  - Total processing time
  - Average processing time per frame

## How It Works

1. **Video Download** (if YouTube URL):
   - Detects if source is a YouTube URL
   - Downloads video using `yt-dlp`
   - Saves to `{script-name}-original-video.mp4` (e.g., `01-person-tracker-with-sam3-original-video.mp4`)
   - Skips download if file already exists

2. **Segment Extraction**:
   - Uses `ffmpeg` to extract each time segment from the original video
   - Creates segment files: `{script-name}-original-video-{from-time}-to-{end-time}.mp4`
   - Segment files are saved in the output directory (or current directory if not specified)

3. **Person Tracking**:
   - Processes each segment with SAM3VideoSemanticPredictor
   - Detects and tracks persons across frames
   - Assigns unique track IDs to each person

4. **Visualization**:
   - Draws segmentation masks
   - Adds tracking trails (last 30 frames)
   - Labels each person with their track ID
   - Displays real-time statistics

5. **Output Generation**:
   - Saves detection results as: `{script-name}-original-video-{from-time}-to-{end-time}-detection-result.mp4`
   - Segment files are kept for reference (not deleted)

## Troubleshooting

### Error: "yt-dlp is required to download YouTube videos"

**Solution**: Install yt-dlp:
```bash
uv pip install yt-dlp
```

### Error: "ffmpeg is required to extract time segments"

**Solution**: Install ffmpeg:
```bash
sudo apt install ffmpeg
```

### Error: "Model file not found"

**Solution**: Ensure the SAM3 model file exists at the specified path:
```bash
ls -la models/sam3.pt
```

### Error: "BPE vocabulary file not found"

**Solution**: Ensure the BPE vocabulary file exists:
```bash
ls -la models/bpe_simple_vocab_16e6.txt.gz
```

### Video Download Fails

- Check your internet connection
- Verify the YouTube URL is valid and accessible
- Some videos may be region-restricted or private

### Low Detection Accuracy

- Try adjusting the `--conf` threshold (lower values = more detections, higher values = fewer but more confident)
- Ensure good video quality
- Check that the video contains clearly visible persons

## Performance Notes

- Processing time depends on:
  - Video resolution
  - Number of persons in frame
  - Hardware (GPU recommended)
- Each segment is processed independently
- Segment files are kept after processing for reference
- Downloaded YouTube videos are cached (not re-downloaded if they exist)
- All files follow a consistent naming convention for easy identification

## Examples

### Process default video with custom confidence:
```bash
./demo/01-person-tracker-with-sam3.sh --conf 0.3
```

### Process custom YouTube video:
```bash
./demo/01-person-tracker-with-sam3.sh --source "https://www.youtube.com/watch?v=ANOTHER_VIDEO"
```

### Process local video file:
```bash
./demo/01-person-tracker-with-sam3.sh --source "/home/user/videos/people.mp4"
```

### Save outputs to specific directory:
```bash
./demo/01-person-tracker-with-sam3.sh --output "results/experiment1.mp4"
```

This will save all files to the `results/` directory with the standard naming convention.

## Notes

- The script processes videos in 3 predefined segments. To change the segments, edit the `SEGMENTS` array in the shell script.
- All files use a consistent naming convention: `{script-name}-original-video-{time-range}[-detection-result].mp4`
- The `--output` argument specifies the output directory. If a file path is provided, the directory is extracted from it.
- Segment files are preserved after processing (not deleted) for reference.
- YouTube videos are cached with the naming convention to avoid re-downloading.
- Each segment generates both a segment file and a detection result file for easier analysis.
- The script uses `uv run` to ensure the correct Python environment is used.

