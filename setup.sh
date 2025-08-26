#!/bin/bash
set -e

# ---- Config ----
ENV_NAME="fmocc"
PYTHON_VERSION="3.10"

# ---- Step 1: Create or update Conda env ----
echo "Creating or updating Conda environment: $ENV_NAME..."
if conda info --envs | grep -q "^$ENV_NAME"; then
    echo "Environment $ENV_NAME exists. Updating..."
    conda update -n $ENV_NAME --all -y
else
    echo "Environment $ENV_NAME does not exist. Creating..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# ---- Step 2: Activate environment ----
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME || {
    echo "❌ Failed to activate environment: $ENV_NAME"
    exit 1
}
echo "✅ Activated environment: $ENV_NAME"

# ---- Step 3: Upgrade pip & build tools ----
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# ---- Step 4: Install fmo-cc in editable mode ----
echo "Installing fmo-cc in editable mode..."
pip install -e . || {
    echo "❌ Failed to install fmo-cc."
    exit 1
}

# ---- Final status ----
echo "✅ FMO-CC setup and run complete!"
echo "Activate the environment with: conda activate $ENV_NAME"
echo "Run manually with: python3 Scripts/run_fmo_cc.py --config input.json"