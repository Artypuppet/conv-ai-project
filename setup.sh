#!/bin/bash
# Setup script for the hallucination detection project.
# Run this once after cloning the repo.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python

if [ ! -d .venv ]; then
  echo "Creating .venv in $SCRIPT_DIR..."
  "$PY" -m venv .venv
fi

echo "Installing Python dependencies (PyTorch CUDA wheels via --extra-index-url in requirements.txt)..."
.venv/bin/python -m pip install -U pip -q
.venv/bin/pip install -r requirements.txt -q

echo "Creating data and output directories..."
mkdir -p data outputs

echo "Setup complete. Activate the environment with: source .venv/bin/activate"
