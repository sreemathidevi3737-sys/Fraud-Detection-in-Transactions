# Fraud Sentinel: Real-Time Credit Card Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fraud Sentinel is a full-stack machine learning application designed to detect fraudulent credit card transactions in real-time. It features a trained Ensemble Random Forest classification model served through a Flask REST API and paired with a responsive, dark-mode React dashboard.

---

## 📌 Architecture Overview

```
Fraud-Detection-in-Transactions/
├── app.py                      # Flask REST API with CORS & inference engine
├── fraud_detection_model.pkl   # Serialized Random Forest / Decision Tree model
├── dataset/
│   └── demo_transactions.csv   # Synthesized benchmark dataset for testing
├── requirements.txt            # Python dependencies (Flask, scikit-learn, CORS, etc.)
├── frontend/                   # React + TypeScript (Vite) client
│   ├── src/
│   │   ├── App.tsx             # Interactive fraud dashboard & risk gauge
│   │   ├── index.css           # Custom dark security dashboard styling
│   │   └── main.tsx            # React application entry point
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

---

## ✨ Features

- **Real-Time Risk Scoring**: Submits transaction attributes ($V1$ through $V28$ PCA features, $Amount$) and computes exact anomaly probabilities.
- **Explainable Anomaly Flags**: Highlights specific feature shifts (e.g., negative shifts in $V14$, $V12$, or extreme velocity spikes in $V4$).
- **One-Click Benchmark Presets**: Auto-populates genuine and fraudulent test cases to easily test model performance without manually entering 28 PCA vectors.
- **Batch CSV Upload Engine**: Evaluates bulk transactional records via CSV uploads for scheduled auditing and anomaly-rate metrics.
- **Clean RESTful API**: Headless architecture allowing custom frontend or mobile client integrations with native CORS support.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ and `npm`

---

### 1. Backend Setup (Flask REST API)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sreemathidevi3737-sys/Fraud-Detection-in-Transactions.git
   cd Fraud-Detection-in-Transactions
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the server:**
   ```bash
   python3 app.py
   ```
   *The API will be live at `http://127.0.0.1:5000`.*

---

### 2. Frontend Setup (React Dashboard)

In a separate terminal window:

1. **Navigate to the frontend folder:**
   ```bash
   cd frontend
   ```

2. **Install node dependencies:**
   ```bash
   npm install
   ```

3. **Start the local Vite development server:**
   ```bash
   npm run dev
   ```
   *Open [http://localhost:5173](http://localhost:5173) in your browser.*

---

## 📡 REST API Reference

### Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "ok",
  "model_type": "RandomForestClassifier",
  "features_count": 30,
  "sample_rows_available": 115
}
```

---

### Load Benchmark Presets
```http
GET /api/samples
```
**Response:**
Returns sample normal and fraud transactions to populate quick-test UI buttons.

---

### Real-Time Transaction Prediction
```http
POST /api/predict
Content-Type: application/json
```

**Payload:**
```json
{
  "Amount": 350.00,
  "V4": 4.85,
  "V11": 3.90,
  "V12": -5.80,
  "V14": -6.90,
  "V17": -5.10
}
```

**Response:**
```json
{
  "success": true,
  "id": 1,
  "amount": 350.0,
  "is_fraud": true,
  "fraud_probability": 0.89,
  "risk_level": "HIGH",
  "flags": [
    "Severe negative shift in V14 transaction pattern",
    "Anomalous V12 deviation detected",
    "Unusually high V4 velocity indicator"
  ]
}
```

---

### Batch File Scoring
```http
POST /api/predict/batch
Content-Type: multipart/form-data
```
- **Body Key:** `file` (a `.csv` file containing transaction rows)
- **Response:** Summary metrics (`total_records`, `flagged_fraud_count`, `fraud_rate`) and row-by-row risk scores.

---

## 🧠 Model Details
- **Architecture**: Ensemble Random Forest / Decision Trees.
- **Input Features**: $V1$ through $V28$ (PCA-transformed latent representations), `Amount`, and `id`.
- **Target**: `Class` (0 = Legitimate, 1 = Fraudulent).

---

## 📄 License
This project is licensed under the MIT License.