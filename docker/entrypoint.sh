#!/usr/bin/env bash
set -e

cd /workspace

# Install robocasa in editable mode when source is mounted (for in-container develop)
if [ -f /workspace/setup.py ]; then
  echo "Installing robocasa from /workspace (editable)..."
  micromamba run -n robocasa pip install -e .
fi

# Kitchen assets: download only when not cached (under robocasa, so volume/rebuild keeps them)
ROBOCASA_ASSETS="${ROBOCASA_ASSETS:-/workspace/robocasa/models/assets}"
if [ -f /workspace/setup.py ]; then
  if [ ! -d "${ROBOCASA_ASSETS}/textures" ] || [ -z "$(ls -A "${ROBOCASA_ASSETS}/textures" 2>/dev/null)" ]; then
    echo "Downloading kitchen assets (~5GB) into robocasa (cached for next runs)..."
    yes | micromamba run -n robocasa python /workspace/robocasa/scripts/download_kitchen_assets.py
  else
    echo "Kitchen assets already present, skipping download."
  fi
fi

# Setup macros once (creates macros_private.py; skip if already exists)
if [ -f /workspace/setup.py ] && ! micromamba run -n robocasa python -c "import robocasa.macros_private" 2>/dev/null; then
  echo "Setting up private macros..."
  micromamba run -n robocasa python /workspace/robocasa/scripts/setup_macros.py
fi

if [ $# -eq 0 ]; then
  exec bash
else
  exec "$@"
fi
