#!/bin/bash
# ==========================================================
# Breast Cancer Detection System - Launcher
# ResNet-50 Deep Learning Prototype
# ==========================================================

echo "=========================================================="
echo " Starting BreastAI Detection Server (ResNet-50 Backend)"
echo "=========================================================="

# Check Python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is required but not found."
    exit 1
fi

# Detect Python interpreter
if [ -f "/Users/adityasuryawanshi/micromamba/bin/python3" ]; then
    PYTHON_CMD="/Users/adityasuryawanshi/micromamba/bin/python3"
else
    PYTHON_CMD="python3"
fi

echo "[INFO] Using Python: $($PYTHON_CMD --version)"

# Check requirements
$PYTHON_CMD -c "import flask, PIL, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[INFO] Installing required dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

echo "[INFO] Starting Flask backend server on http://127.0.0.1:5050..."
echo "[INFO] Press Ctrl+C to stop the server."
echo ""

# Auto-open browser on macOS after short delay
(sleep 1.2 && open http://127.0.0.1:5050) &

$PYTHON_CMD app.py
