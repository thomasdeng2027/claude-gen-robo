#!/usr/bin/env bash
# setup.sh — create and populate the claude-data-gen conda environment.
#
# Run once:
#   bash setup.sh
#
# The script is idempotent: re-running it will reinstall packages into an
# existing env without deleting prior work.

set -e

# Redirect pip cache off AFS (quota-limited) onto local disk
export PIP_CACHE_DIR=/scr/jingyuny/pip-cache
mkdir -p "$PIP_CACHE_DIR"

CONDA_ROOT=/juno/u/jingyuny/.miniconda3
ENV_NAME=claude-data-gen
PYTHON_VERSION=3.11          # must match Isaac Sim 5.1 requirement
CUROBO_ROOT=/juno/u/jingyuny/curobo

CONDA=$CONDA_ROOT/bin/conda
PIP=$CONDA_ROOT/envs/$ENV_NAME/bin/pip
PYTHON=$CONDA_ROOT/envs/$ENV_NAME/bin/python

# ── 1. Create env ─────────────────────────────────────────────────────────────
if $CONDA env list | grep -q "^$ENV_NAME "; then
    echo "[setup] Env '$ENV_NAME' already exists — skipping creation."
else
    echo "[setup] Creating conda env '$ENV_NAME' (Python $PYTHON_VERSION)..."
    $CONDA create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

# ── 2. Isaac Sim 5.1 ──────────────────────────────────────────────────────────
# 5.1.0.0 ships warp 1.8.2 which fixes sm_120 (Blackwell/RTX 5090) PTX compilation
# and the cuDeviceGetUuid ABI issue with CUDA driver 13.0 (both regressions in 5.0.0.0).
echo "[setup] Installing Isaac Sim 5.1.0.0..."
$PIP install \
    isaacsim==5.1.0.0 \
    isaacsim-rl==5.1.0.0 \
    isaacsim-replicator==5.1.0.0 \
    isaacsim-extscache-physics==5.1.0.0 \
    isaacsim-extscache-kit==5.1.0.0 \
    isaacsim-extscache-kit-sdk==5.1.0.0 \
    --extra-index-url https://pypi.nvidia.com

# ── 3. Accept Omniverse EULA (required before first Isaac Sim import) ─────────
echo "[setup] Accepting Omniverse EULA..."
echo "Yes" > "$CONDA_ROOT/envs/$ENV_NAME/lib/python3.11/site-packages/omni/EULA_ACCEPTED"

# ── 4. cuRobo — add to sys.path via .pth file (editable install is broken) ────
# pip install -e fails because curobo's pyproject.toml references a src/images
# directory that doesn't exist. Replicate what isaac5 does: drop a .pth file.
echo "[setup] Adding cuRobo to sys.path via .pth file..."
SITE_PACKAGES=$CONDA_ROOT/envs/$ENV_NAME/lib/python3.11/site-packages
echo "$CUROBO_ROOT/src" > "$SITE_PACKAGES/curobo.pth"
echo "  -> $CUROBO_ROOT/src added to sys.path"

# ── 5. Pipeline + cuRobo runtime dependencies ─────────────────────────────────
# typing_extensions: isaacsim-kernel pins ==4.12.2 but anthropic needs >=4.14;
#   4.15 works fine at runtime with Isaac Sim despite the metadata conflict.
# websockets: isaacsim-kernel requires ==12.0; viser wants >=13.1 but viser is
#   only used for curobo visualization which we don't use — Isaac Sim wins.
echo "[setup] Installing pipeline and cuRobo dependencies..."
$PIP install \
    anthropic loguru numpy \
    setuptools_scm yourdfpy "warp-lang>=0.10.0" \
    importlib_resources tqdm viser \
    "typing_extensions==4.15.0" \
    "websockets==12.0"

echo ""
echo "[setup] Done. Activate with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "Then run the pipeline:"
echo "    cd $(dirname $0)"
echo "    python src/pipeline_parallel.py --tasks bottle_lift --n-envs 4"
echo ""
echo "Or the grasp debug script:"
echo "    python src/debug_grasp_close.py"
