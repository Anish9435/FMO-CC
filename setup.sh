#!/bin/bash
set -e

# ---- Config ----
ENV_DIR="fmocc_env"
PYTHON_EXEC="python3"           # Default Python executable
MIN_PYTHON_VERSION="3.8"

# ---- Step 0: Check Python version ----
PYTHON_VERSION=$($PYTHON_EXEC -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
PYTHON_MAJOR=$($PYTHON_EXEC -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$($PYTHON_EXEC -c 'import sys; print(sys.version_info[1])')

if (( PYTHON_MAJOR < 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR < 8) )); then
    echo "❌ Python $MIN_PYTHON_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python version $PYTHON_VERSION detected."

# ---- Step 1: Create virtual environment if it doesn't exist ----
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment in $ENV_DIR..."
    $PYTHON_EXEC -m venv "$ENV_DIR"
else
    echo "Virtual environment $ENV_DIR already exists."
fi

# ---- Step 2: Activate the environment ----
echo "Activating virtual environment..."
source "$ENV_DIR/bin/activate" || {
    echo "❌ Failed to activate virtual environment."
    exit 1
}
echo "✅ Activated virtual environment."

# ---- Step 3: Upgrade pip, setuptools, wheel ----
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# ---- Step 4: Install fmo-cc in editable mode ----
echo "Installing fmo-cc in editable mode..."
pip install -e . || {
    echo "❌ Failed to install fmo-cc."
    exit 1
}

# ---- Step 5: Optional extra dependencies ----
# echo "Installing additional dependencies (if any)..."
# pip install -r requirements.txt

# ---- Step 6: Cleanup old pycache (optional) ----
echo "Cleaning up __pycache__ directories..."
find src/fmocc -type d -name "__pycache__" -exec rm -rf {} +

# ---- Final status ----
echo "✅ FMO-CC setup complete!"
echo "Activate environment with: source $ENV_DIR/bin/activate"