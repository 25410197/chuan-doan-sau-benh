# Pepper Disease Detection & Classification

A deep learning project for automated detection and classification of pepper plant diseases using TensorFlow/Keras and MobileNetV2.

## Installation

### Prerequisites
- Python 3.10.11
- pip or conda

### Setup Steps

1. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model

```bash
python train.py
```

**What it does:**
- Loads training data from `dataset/train/`
- Loads validation data from `dataset/val/`
- Applies data augmentation and normalization
- Trains MobileNetV2 with balanced class weights
- Saves model as `pepper_disease_model.keras`


### 2. Start FastAPI Server (Production)

```bash
python main.py
```

Then run in another terminal:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

- **GET `/`** - Health check
  ```bash
  curl http://localhost:8000/
  ```

- **POST `/predict`** - Make predictions
  ```bash
  curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/predict
  ```