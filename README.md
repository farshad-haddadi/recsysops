# RecSysOps

A production-style Recommendation System + MLOps project built with Matrix Factorization, FastAPI, Docker, and GitHub Actions.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-success)

---

# Overview

RecSysOps is an end-to-end recommendation system built on the MovieLens 100K dataset.

The project demonstrates:

- Recommendation model training
- Matrix Factorization implementation
- Model persistence
- Experiment tracking
- FastAPI serving layer
- Docker containerization
- Automated testing
- CI/CD with GitHub Actions

The goal is to simulate how a recommendation model would be developed, evaluated, deployed, and maintained in a real-world MLOps environment.

---

# Architecture

```text
MovieLens 100K Dataset
          |
          v
Training Pipeline
          |
          v
Matrix Factorization Model
          |
          v
Model Artifact (.pkl)
          |
          v
Model Registry
          |
          v
FastAPI Service
    ├── /health
    ├── /recommend
    ├── /model-info
    └── /metrics
```

---

# Project Structure

```text
recsysops/
│
├── app/
│   ├── api/
│   ├── core/
│   ├── schemas/
│   └── main.py
│
├── training/
│
├── inference/
│
├── tests/
│
├── artifacts/
│   ├── experiments/
│   └── models/
│
├── docker/
│
├── configs/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── requirements.txt
│
└── README.md
```

---

# Dataset

This project uses the MovieLens 100K dataset.

Dataset statistics:

- Users: 943
- Movies: 1,682
- Ratings: 100,000

Source:

https://grouplens.org/datasets/movielens/

---

# Model

Algorithm:

- Matrix Factorization

Model parameters:

- Latent Factors: 20
- Learning Rate: 0.01
- Regularization: 0.02
- Epochs: 10

The trained model is persisted as:

```text
artifacts/models/matrix_factorization.pkl
```

---

# Experiment Tracking

Training runs automatically generate experiment metadata.

Stored in:

```text
artifacts/experiments/
```

Example:

```json
{
  "model_name": "matrix_factorization",
  "timestamp_utc": "20260604_202645",
  "params": {
    "num_factors": 20,
    "learning_rate": 0.01
  },
  "metrics": {
    "precision_at_k": 0.0036,
    "recall_at_k": 0.0361
  }
}
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/farshad-haddadi/recsysops.git
cd recsysops
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate:

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Train Model

Run training pipeline:

```bash
python training/train_matrix_factorization.py
```

This generates:

```text
artifacts/models/matrix_factorization.pkl
artifacts/experiments/*.json
```

---

# Run API

Start FastAPI server:

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

# API Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

## Recommendations

```http
POST /recommend
```

Request:

```json
{
  "user_id": 1,
  "k": 10
}
```

Response:

```json
{
  "user_id": 1,
  "model_name": "matrix_factorization",
  "recommendations": [
    {
      "rank": 1,
      "item_id": 50,
      "title": "Star Wars (1977)",
      "score": 4.12
    }
  ]
}
```

---

## Model Information

```http
GET /model-info
```

Response:

```json
{
  "model_name": "matrix_factorization",
  "model_path": "artifacts/models/matrix_factorization.pkl",
  "num_users": 943,
  "num_items": 1679,
  "num_factors": 20
}
```

---

## Metrics

```http
GET /metrics
```

Response:

```json
{
  "model_name": "matrix_factorization",
  "metrics": {
    "precision_at_k": 0.0036,
    "recall_at_k": 0.0361
  }
}
```

---

# Running Tests

Run all tests:

```bash
python -m pytest
```

Expected:

```text
13 passed
```

---

# Docker

Build and start:

```bash
docker compose up --build
```

API available at:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

Stop containers:

```bash
docker compose down
```

---

# CI/CD

GitHub Actions automatically:

- Installs dependencies
- Runs tests
- Validates pull requests
- Validates pushes to main

Workflow file:

```text
.github/workflows/tests.yml
```

---

# Results

Latest evaluation metrics:

| Metric | Value |
|----------|----------|
| Precision@10 | 0.0036 |
| Recall@10 | 0.0361 |
| Users Evaluated | 943 |

---

# Future Improvements

Possible next steps:

- Hyperparameter tuning
- MLflow integration
- Prometheus monitoring
- Kubernetes deployment
- Feature store integration
- Implicit feedback models
- Neural collaborative filtering
- A/B testing framework

---

# Tech Stack

- Python 3.11
- FastAPI
- NumPy
- Pandas
- PyTest
- Docker
- GitHub Actions

---

# Author

Farshad Haddadi

GitHub:

https://github.com/farshad-haddadi

---

# License

This project is for educational and portfolio purposes.