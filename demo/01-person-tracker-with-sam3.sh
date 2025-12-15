#!/bin/bash
# Shell wrapper script for 01-person-tracker-with-sam3.py
# Uses uv run to execute the script in the virtual environment
# Handles YouTube URL downloads before passing to Python script

set -e  # Exit on error

# Default YouTube URL (used if --source is not specified)
DEFAULT_YOUTUBE_URL="https://www.youtube.com/watch?v=qjSUk9o-D6E&t=1s"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Get the Python script path
PYTHON_SCRIPT="$SCRIPT_DIR/01-person-tracker-with-sam3.py"

# Get script name without extension for file naming
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Make sure Python script is executable
chmod +x "$PYTHON_SCRIPT"

# Function to check if a string is a YouTube URL
is_youtube_url() {
    local url="$1"
    if [[ "$url" =~ ^https?://(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com) ]]; then
        return 0
    fi
    # Also check for youtu.be short URLs
    if [[ "$url" =~ ^https?://youtu\.be/ ]]; then
        return 0
    fi
    return 1
}

# Function to extract video ID from YouTube URL
extract_video_id() {
    local url="$1"
    local video_id=""
    
    # Try youtu.be format first
    if [[ "$url" =~ youtu\.be/([^?&]+) ]]; then
        video_id="${BASH_REMATCH[1]}"
    # Try youtube.com format with v= parameter
    elif [[ "$url" =~ \?v=([^&]+) ]] || [[ "$url" =~ \&v=([^&]+) ]]; then
        video_id="${BASH_REMATCH[1]}"
    fi
    
    echo "$video_id"
}

# Function to download YouTube video
download_youtube_video() {
    local url="$1"
    local output_dir="$2"
    local script_name="$3"
    
    # Extract video ID
    local video_id=$(extract_video_id "$url")
    
    if [ -z "$video_id" ]; then
        echo "Error: Could not extract video ID from URL: $url"
        exit 1
    fi
    
    local output_path="$output_dir/${script_name}-original-video.mp4"
    
    # If video already exists, return it
    if [ -f "$output_path" ]; then
        echo "Video already exists at $output_path, skipping download." >&2
        echo "$output_path"
        return 0
    fi
    
    echo "Downloading video from YouTube: $url" >&2
    echo "Output will be saved to: $output_path" >&2
    
    # Check if yt-dlp is available
    if command -v yt-dlp &> /dev/null; then
        YT_DLP_CMD="yt-dlp"
    elif uv run --no-project python -m yt_dlp --version &> /dev/null; then
        YT_DLP_CMD="uv run --no-project python -m yt_dlp"
    else
        echo "Error: yt-dlp is required to download YouTube videos." >&2
        echo "Install it with: uv pip install yt-dlp" >&2
        exit 1
    fi
    
    # Download the video (redirect yt-dlp output to stderr to keep stdout clean)
    $YT_DLP_CMD \
        --format 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' \
        --output "${output_path%.mp4}" \
        --merge-output-format mp4 \
        "$url" >&2
    
    # Check if download was successful
    if [ -f "$output_path" ]; then
        echo "Video downloaded successfully to: $output_path" >&2
        echo "$output_path"
    else
        # Try to find the actual downloaded file (yt-dlp might have added extension or used different name)
        local found_file=$(find "$output_dir" -name "${script_name}-original-video*" -type f | head -n 1)
        if [ -n "$found_file" ]; then
            echo "Video downloaded successfully to: $found_file" >&2
            echo "$found_file"
        else
            echo "Error: Downloaded file not found at expected location: $output_path" >&2
            exit 1
        fi
    fi
}

# Function to extract time segment from video using ffmpeg
# This function slices segments from the downloaded original video
extract_time_segment() {
    local input_video="$1"  # The downloaded original video
    local start_time="$2"    # Start time in MM:SS format
    local end_time="$3"      # End time in MM:SS format
    local output_path="$4"   # Output segment file path
    
    # Check if ffmpeg is available
    if ! command -v ffmpeg &> /dev/null; then
        echo "Error: ffmpeg is required to extract time segments."
        echo "Install it with: sudo apt install ffmpeg"
        exit 1
    fi
    
    # Convert times to seconds and calculate duration
    local start_sec=$(time_to_seconds "$start_time")
    local end_sec=$(time_to_seconds "$end_time")
    local duration=$((end_sec - start_sec))
    
    echo "Slicing segment from original video: $start_time - $end_time (duration: ${duration}s)" >&2
    echo "Input video: $input_video" >&2
    echo "Output segment: $output_path" >&2
    
    # Detect video codec using ffprobe
    local codec=""
    if command -v ffprobe &> /dev/null; then
        codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$input_video" 2>/dev/null | head -1 | tr -d '[:space:]')
    fi
    
    # Extract the segment using ffmpeg
    # For AV1 codec, we need to re-encode to ensure proper headers (AV1 with copy has missing sequence headers)
    # For other codecs, use stream copy (fast, no re-encoding)
    if [ -n "$codec" ] && ([ "$codec" = "av01" ] || [ "$codec" = "av1" ]); then
        echo "Detected AV1 codec - re-encoding to ensure compatibility..." >&2
        # Re-encode AV1 to H.264 for compatibility (AV1 segments with copy can have missing headers)
        ffmpeg -i "$input_video" \
            -ss "$start_time" \
            -t "$duration" \
            -c:v libx264 \
            -preset fast \
            -crf 23 \
            -c:a aac \
            -b:a 128k \
            -avoid_negative_ts make_zero \
            -y \
            "$output_path" >&2
    else
        if [ -n "$codec" ]; then
            echo "Using stream copy (fast mode) for codec: $codec" >&2
        else
            echo "Codec detection failed, using stream copy (fast mode)" >&2
        fi
        # Use stream copy for other codecs (fast, preserves original codec)
        ffmpeg -i "$input_video" \
            -ss "$start_time" \
            -t "$duration" \
            -c copy \
            -avoid_negative_ts make_zero \
            -y \
            "$output_path" >&2
    fi
    
    if [ ! -f "$output_path" ]; then
        echo "Error: Failed to extract segment to $output_path" >&2
        exit 1
    fi
    
    echo "Segment extracted to: $output_path" >&2
}

# Function to convert time string (MM:SS) to seconds
time_to_seconds() {
    local time_str="$1"
    local minutes=$(echo "$time_str" | cut -d: -f1 | sed 's/^0*//')
    local seconds=$(echo "$time_str" | cut -d: -f2 | sed 's/^0*//')
    # Handle empty values (leading zeros removed)
    [ -z "$minutes" ] && minutes=0
    [ -z "$seconds" ] && seconds=0
    echo "$((minutes * 60 + seconds))"
}

# Process arguments to find --source and handle YouTube URLs
SOURCE_VIDEO=""
ARGS=()
i=1

while [ $i -le $# ]; do
    arg="${!i}"
    
    if [ "$arg" = "--source" ] && [ $i -lt $# ]; then
        next_arg="${!((i + 1))}"
        
        # Check if it's a YouTube URL
        if is_youtube_url "$next_arg"; then
            echo "Detected YouTube URL, downloading video..."
            SOURCE_VIDEO=$(download_youtube_video "$next_arg" "$SCRIPT_DIR" "$SCRIPT_NAME")
        else
            # Not a YouTube URL, use as-is
            SOURCE_VIDEO="$next_arg"
        fi
        
        # Check if source video exists
        if [ ! -f "$SOURCE_VIDEO" ]; then
            echo "Error: Source video file not found: $SOURCE_VIDEO"
            exit 1
        fi
        
        i=$((i + 2))  # Skip both --source and its value
        continue
    fi
    
    # Store other arguments (but skip --output as we'll generate segment-specific outputs)
    if [ "$arg" = "--output" ] && [ $i -lt $# ]; then
        # Skip --output and its value (we'll generate segment-specific outputs)
        i=$((i + 2))
        continue
    fi
    
    ARGS+=("$arg")
    
    i=$((i + 1))
done

# Check if we have a source video, use default YouTube URL if not specified
if [ -z "$SOURCE_VIDEO" ]; then
    echo "No --source specified, using default YouTube URL: $DEFAULT_YOUTUBE_URL"
    if is_youtube_url "$DEFAULT_YOUTUBE_URL"; then
        SOURCE_VIDEO=$(download_youtube_video "$DEFAULT_YOUTUBE_URL" "$SCRIPT_DIR" "$SCRIPT_NAME")
    else
        echo "Error: Default URL is not a valid YouTube URL: $DEFAULT_YOUTUBE_URL"
        exit 1
    fi
fi

# Define time segments to process
# Format: "start_time end_time" (in MM:SS format)
SEGMENTS=(
    "00:26 00:42"
    "01:00 01:24"
    "01:48 02:32"
)

# All files should be saved in the demo folder
OUTPUT_DIR="$SCRIPT_DIR"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Process each segment
for segment in "${SEGMENTS[@]}"; do
    START_TIME=$(echo "$segment" | cut -d' ' -f1)
    END_TIME=$(echo "$segment" | cut -d' ' -f2)
    
    # Format time for filename (replace : with -)
    START_TIME_FORMATTED="${START_TIME//:/-}"
    END_TIME_FORMATTED="${END_TIME//:/-}"
    
    echo ""
    echo "========================================="
    echo "Processing segment: $START_TIME - $END_TIME"
    echo "========================================="
    
    # Create segment video file name: {script-name}-original-video-{from-to-time}.mp4
    SEGMENT_VIDEO="$OUTPUT_DIR/${SCRIPT_NAME}-original-video-${START_TIME_FORMATTED}-to-${END_TIME_FORMATTED}.mp4"
    
    # Result file name: {script-name}-original-video-{from-to-time}-detection-result.mp4
    SEGMENT_OUTPUT="$OUTPUT_DIR/${SCRIPT_NAME}-original-video-${START_TIME_FORMATTED}-to-${END_TIME_FORMATTED}-detection-result.mp4"
    
    # Extract the time segment from the downloaded video using ffmpeg
    # SOURCE_VIDEO is the downloaded video: {script-name}-original-video.mp4
    extract_time_segment "$SOURCE_VIDEO" "$START_TIME" "$END_TIME" "$SEGMENT_VIDEO"
    
    # Validate the segment file was created and is readable
    if [ ! -f "$SEGMENT_VIDEO" ]; then
        echo "Error: Segment file was not created: $SEGMENT_VIDEO" >&2
        continue
    fi
    
    # Check if the segment file is valid using ffprobe
    if command -v ffprobe &> /dev/null; then
        if ! ffprobe -v error "$SEGMENT_VIDEO" &>/dev/null; then
            echo "Error: Segment file appears to be corrupted: $SEGMENT_VIDEO" >&2
            continue
        fi
    fi
    
    # Build arguments for this segment
    SEGMENT_ARGS=("${ARGS[@]}")
    SEGMENT_ARGS+=("--source" "$SEGMENT_VIDEO")
    SEGMENT_ARGS+=("--output" "$SEGMENT_OUTPUT")
    
    # Run the Python script for this segment
    echo "Processing segment with SAM3..."
    uv run --no-project python "$PYTHON_SCRIPT" "${SEGMENT_ARGS[@]}"
    
    # Note: Segment video file is kept (not deleted) as per new naming convention
done

echo ""
echo "========================================="
echo "All segments processed successfully!"
echo "========================================="
