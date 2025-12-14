# Cursor IDE Configuration

This file provides instructions for Cursor IDE to ensure consistent Python dependency management using `uv`.

## Python Environment Management

### Virtual Environments

- **Always use `uv venv`** to create virtual environments
- Default virtual environment location: `.venv` in the project root
- Never use `python -m venv`, `virtualenv`, or `conda` for creating virtual environments unless explicitly requested

### Dependency Management

- **Always use `uv sync`** to install dependencies from `pyproject.toml`
- **Always use `uv pip install`** when installing packages (instead of plain `pip install`)
- **Always use `pyproject.toml`** (or other TOML files) for Python dependency management
- Never use `pip install`, `pipenv`, `poetry`, or other package managers unless explicitly requested by the user

### Package Installation Commands

When installing Python packages:
- Use `uv pip install <package>` instead of `pip install <package>`
- Use `uv sync` to install all dependencies from `pyproject.toml`
- For editable installs, use `uv pip install -e <path>`

### PyTorch and GPU Packages

When installing PyTorch or GPU-related packages:
- Always ensure CUDA support is included
- Use the appropriate PyTorch index URL for CUDA versions (e.g., `--index-url https://download.pytorch.org/whl/cu121`)
- Verify GPU availability after installation when relevant

### Virtual Environment Activation

When providing commands or scripts:
- Always activate the virtual environment first: `source .venv/bin/activate`
- Include activation commands in any installation or setup scripts
- Assume the virtual environment is at `.venv` unless specified otherwise

## Project-Specific Notes

- This project uses `uv` for all Python package management
- The virtual environment is located at `.venv/`
- Dependencies are managed through `pyproject.toml`
- Ultralytics is installed from the local submodule in editable mode
- GPU support is required (NVIDIA L40 GPUs available)

## Examples

### Correct Usage

```bash
# Create virtual environment
uv venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv sync

# Install a package
uv pip install numpy

# Install from submodule in editable mode
uv pip install -e submodules/ultralytics
```

### Incorrect Usage (Avoid)

```bash
# Don't use plain pip
pip install numpy

# Don't use python -m venv
python -m venv venv

# Don't use pipenv or poetry
pipenv install
poetry install
```

