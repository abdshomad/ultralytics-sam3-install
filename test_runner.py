#!/usr/bin/env python3
"""
Utility script to execute commands and save output as PNG screenshots.
"""
import subprocess
import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

def get_font(size=14):
    """Get a monospace font for rendering."""
    try:
        # Try to use a common monospace font
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
    except:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", size)
        except:
            # Fallback to default font
            return ImageFont.load_default()

def text_to_image(text, output_path, width=1200, padding=20, bg_color=(20, 20, 20), text_color=(200, 200, 200)):
    """Convert text output to an image and save as PNG."""
    lines = text.split('\n')
    
    # Calculate dimensions
    font = get_font(14)
    line_height = 20
    max_lines = min(len(lines), 100)  # Limit to 100 lines to avoid huge images
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    
    height = len(lines) * line_height + 2 * padding
    
    # Create image
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw text
    y = padding
    for line in lines:
        # Truncate long lines
        if len(line) > 140:
            line = line[:137] + "..."
        draw.text((padding, y), line, fill=text_color, font=font)
        y += line_height
    
    # Save image
    img.save(output_path, 'PNG')
    print(f"Saved screenshot to {output_path}")

def execute_command(cmd, cwd=None, env=None, timeout=300):
    """Execute a command and capture output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = f"Command: {cmd}\n"
        output += f"Exit code: {result.returncode}\n"
        output += f"\n--- STDOUT ---\n{result.stdout}\n"
        if result.stderr:
            output += f"\n--- STDERR ---\n{result.stderr}\n"
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return f"Command: {cmd}\n\nERROR: Command timed out after {timeout} seconds\n", 1
    except Exception as e:
        return f"Command: {cmd}\n\nERROR: {str(e)}\n", 1

def execute_python_script(script_content, cwd=None, env=None, timeout=300):
    """Execute a Python script and capture output."""
    try:
        # Write script to temporary file
        script_path = Path("/tmp/test_script.py")
        script_path.write_text(script_content)
        
        # Execute script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = f"Python Script Execution\n"
        output += f"Exit code: {result.returncode}\n"
        output += f"\n--- STDOUT ---\n{result.stdout}\n"
        if result.stderr:
            output += f"\n--- STDERR ---\n{result.stderr}\n"
        
        # Clean up
        script_path.unlink()
        
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return f"Python Script Execution\n\nERROR: Script timed out after {timeout} seconds\n", 1
    except Exception as e:
        return f"Python Script Execution\n\nERROR: {str(e)}\n", 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: test_runner.py <command|python> <output_path> [command_or_script]")
        sys.exit(1)
    
    mode = sys.argv[1]
    output_path = sys.argv[2]
    
    if mode == "command":
        if len(sys.argv) < 4:
            print("Error: Command required for 'command' mode")
            sys.exit(1)
        cmd = sys.argv[3]
        output, exit_code = execute_command(cmd)
    elif mode == "python":
        if len(sys.argv) < 4:
            print("Error: Python script content required for 'python' mode")
            sys.exit(1)
        script_content = sys.argv[3]
        output, exit_code = execute_python_script(script_content)
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'command' or 'python'")
        sys.exit(1)
    
    text_to_image(output, output_path)
    sys.exit(exit_code)
