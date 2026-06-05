# RecSysOps

A production-style Recommendation System + MLOps project built with FastAPI, PyTorch, Docker, GitHub Actions, Matrix Factorization, and Two-Tower Retrieval.

## Overview

RecSysOps is an end-to-end recommendation system designed to demonstrate how machine learning models can be trained, tracked, tested, deployed, and served through a production-style API.

The project supports multiple recommendation algorithms and allows runtime model selection through a FastAPI service.

Built using the MovieLens 100K dataset.

---

## Features

### Recommendation Models

#### Matrix Factorization

* Collaborative filtering recommender
* User-item latent embeddings
* Model persistence using Pickle
* Fast recommendation generation

#### Two-Tower Retrieval Model

* Neural recommendation architecture implemented in PyTorch
* Separate user and item embedding towers
* Negative sampling training pipeline
* Model checkpoint persistence
* Experiment tracking integration

---

## MLOps Features

### Experiment Tracking

Training runs automatically generate experiment artifacts containing:

* Model name
* Hyperparameters
* Training metrics
* Timestamps

Example:

```json
{
  "model_name": "two_tower",
  "params": {
    "embedding_dim": 64,
    "epochs": 5,
    "learning_rate": 0.001
  },
  "metrics": {
    "final_loss": 0.3782
  }
}
```

Stored in:

```text
artifacts/experiments/
```

---

### Model Persistence

Saved model artifacts:

```text
artifacts/models/
├── matrix_factorization.pkl
└── two_tower.pt
```

---

### Automated Testing

The project includes automated unit and integration tests covering:

* API endpoints
* Model loading
* Recommendation generation
* Metrics endpoint
* Dataset creation
* Two-Tower model
* Experiment tracking

Current status:

```text
20+ passing tests
```

---

### CI/CD

GitHub Actions automatically runs the test suite on every push.

Pipeline includes:

* Dependency installation
* Test execution
* Validation checks

---

### Docker Support

Application can be containerized and executed consistently across environments.

```bash
docker build -t recsysops .
docker run -p 8000:8000 recsysops
```

---

## API Endpoints

### Generate Recommendations

```http
POST /recommend
```

Example request:

```json
{
  "user_id": 1,
  "k": 10,
  "model_name": "matrix_factorization"
}
```

Supported models:

* matrix_factorization
* two_tower

---

### Model Information

```http
GET /model-info
```

Returns metadata about available models.

---

### Compare Models

```http
GET /compare-models
```

Returns a comparison of supported recommendation models.

---

### Metrics

```http
GET /metrics
```

Returns operational metrics.

---

### Health Check

```http
GET /health
```

Returns service health status.

---

## Project Structure

```text
recsysops/
│
├── app/
│   ├── api/
│   ├── core/
│   ├── routes/
│   └── main.py
│
├── training/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── experiment_tracking/
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
├── .github/workflows/
│
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Local Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run API:

```bash
python -m uvicorn app.main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

## Training

### Matrix Factorization

```bash
python training/train_matrix_factorization.py
```

### Two-Tower Retrieval

```bash
python training/train_two_tower.py
```

Training automatically:

* Saves model artifacts
* Logs experiment results
* Generates reproducible outputs

---

## Technology Stack

### Machine Learning

* PyTorch
* NumPy
* Pandas

### Backend

* FastAPI
* Pydantic
* Uvicorn

### MLOps

* Docker
* GitHub Actions
* Experiment Tracking

### Testing

* Pytest

### Dataset

* MovieLens 100K

---

## Architecture

```text
MovieLens Dataset
        │
        ▼
 Training Pipelines
        │
        ├── Matrix Factorization
        │
        └── Two-Tower Retrieval
        │
        ▼
 Model Artifacts
        │
        ▼
 FastAPI Inference Service
        │
        ▼
 Recommendation API
```

---

## Author

Farshad Haddadi

GitHub:
https://github.com/farshad-haddadi
