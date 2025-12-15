#!/bin/bash
# Shell wrapper script for 01-person-tracker-with-sam3.py
# Uses uv run to execute the script in the virtual environment

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Get the Python script path
PYTHON_SCRIPT="$SCRIPT_DIR/01-person-tracker-with-sam3.py"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Make sure Python script is executable
chmod +x "$PYTHON_SCRIPT"

# Run the Python script using uv run (automatically uses .venv)
# Use --no-project to avoid building the project package
uv run --no-project python "$PYTHON_SCRIPT" "$@"
