#!/bin/bash
set -e

# Sync sentinel for setup.sh (IsaacGym workflow). Harmless when no setup.sh
# is reading it — just an empty file in /tmp that gets touched at end.
rm -f /tmp/entrypoint_done

export PATH="/opt/venv/bin:/usr/local/bin:${PATH:-/usr/bin:/bin}"
export VIRTUAL_ENV="/opt/venv"

# ── 1. Editable install (project mounted at /workspace/robocasa) ─────
# Both branches resolve install_requires by default. If you need --no-deps
# (e.g. to avoid uv re-resolving heavy science stack), add a `post_install_hooks`
# entry to install_plan.json that re-runs the install with --no-deps.
if [ -f "/workspace/robocasa/pyproject.toml" ]; then
    echo ">> Installing editable package (pyproject.toml)..."
    cd /workspace/robocasa && uv pip install -e . --index-strategy unsafe-best-match && cd - > /dev/null
elif [ -f "/workspace/robocasa/setup.py" ]; then
    echo ">> Installing editable package (setup.py)..."
    cd /workspace/robocasa && uv pip install -e . --index-strategy unsafe-best-match && cd - > /dev/null
fi

# ── 2. Post-install hooks from InstallationPlan ──────────────────────────────
# Rendered by render_base.py from <repo>/.nautilus/install_plan.json's
# `post_install_hooks`. `when=first_run` entries are wrapped in a sentinel
# guard; `when=every_run` entries fire on every container start.
if ([ -f /workspace/robocasa/pyproject.toml ] || [ -f /workspace/robocasa/setup.py ]) && ! python -c 'import robocasa.macros_private' 2>/dev/null; then echo 'Setting up private macros...'; python /workspace/robocasa/scripts/setup_macros.py; fi
ROBOCASA_ASSETS="${ROBOCASA_ASSETS:-/workspace/robocasa/robocasa/models/assets}"; if [ ! -d "${ROBOCASA_ASSETS}/textures" ] || [ -z "$(ls -A "${ROBOCASA_ASSETS}/textures" 2>/dev/null)" ]; then echo 'Downloading kitchen assets (~5GB)...'; yes | python /workspace/robocasa/robocasa/scripts/download_kitchen_assets.py; else echo 'Kitchen assets already present, skipping.'; fi

# 
# Slot for downstream sub-skills to inject project-specific steps.

# <<<EXTENSION_ENTRYPOINT_INSERT_ABOVE>>> — sub-skills insert pre-exec hooks above this line

echo ">> Ready."
touch /tmp/entrypoint_done
exec "$@"
