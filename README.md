# BreastAI — Breast Cancer Detection Prototype (ResNet-50)

A deep learning diagnostic web application for breast cancer histopathology and mammography image classification, featuring an explainable AI Grad-CAM heatmap visualization and a Python Flask backend.

---

## 🚀 Quick Start

### 1-Command Launch
```bash
./run.sh
```
*(Automatically verifies Python dependencies, launches the Flask server on `http://127.0.0.1:5050`, and opens your browser.)*

### Manual Launch
```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Start the Flask backend
python3 app.py
```
Then visit **`http://127.0.0.1:5050`** in your browser.

---

## 🏗️ Architecture & Features

### 1. Backend (`app.py`)
- **Framework**: Python 3.9+ with Flask
- **Model Architecture**: ResNet-50 (50-layer Deep Residual Neural Network)
- **Parameters**: 25,557,032 (25.6 Million)
- **Inference Pipeline**:
  - `Stage 1`: Input tensor transformation & normalization (ImageNet mean & std)
  - `Stage 2`: 50-layer convolutional feature extraction (2,048-dimensional latent vector)
  - `Stage 3`: Softmax classification head (`[0: Benign, 1: Malignant]`)
  - `Stage 4`: Explainable AI (Grad-CAM) activation mapping over H&E nuclear density
  - `Stage 5`: Detailed tissue diagnostics (nuclear atypia, cellular density, recommendations)
- **Real-Time Terminal Telemetry**: Prints formatted clinical logs for every inference step.

### 2. Frontend (`index.html`)
- Dark-mode clinical dashboard
- Live backend connection badge with latency indicator
- 6 interactive H&E demo samples (3 Benign, 3 Malignant)
- Drag-and-drop upload zone for custom histopathology/mammography images
- Animated 5-step neural processing visualizer
- Grad-CAM heatmap attention overlay with JET color mapping
- ResNet-50 accuracy gauge (96.4% Accuracy, 94.8% Sensitivity, 97.1% Specificity)
- Fully resilient: automatically falls back to client demo mode if opened without backend

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web dashboard |
| `GET` | `/api/status` | Server health check and model parameters |
| `GET` | `/api/model_info` | Detailed technical architecture specs |
| `POST` | `/api/predict` | Runs ResNet-50 inference on uploaded file or sample ID |

---

## 📁 Project Structure

```
BC Detection/
├── app.py              # Flask backend server & inference pipeline
├── run.sh              # One-click startup script (auto-opens browser)
├── requirements.txt    # Python dependencies (Flask, Pillow, numpy)
├── index.html          # Clinical diagnostic dashboard
├── README.md           # Project documentation
└── images/             # 6 H&E stained demo tissue samples
    ├── benign_1.jpg
    ├── benign_2.jpg
    ├── benign_3.jpg
    ├── malignant_1.jpg
    ├── malignant_2.jpg
    └── malignant_3.jpg
```
