# RecSysOps
### Production-Ready Recommendation System

> 🚧 **Status: Under Active Development**
>
> This project is currently being built incrementally as part of a deep dive into recommendation systems, ML engineering, and production ML infrastructure. New features, models, evaluations, and deployment components are being added over time.

---

## Overview

RecSysOps is an end-to-end recommendation system designed to demonstrate modern machine learning engineering practices beyond notebook-based experimentation.

The project explores how recommendation models are developed, evaluated, deployed, and maintained in production environments.

The system will evolve from simple recommendation baselines to a scalable two-tower neural retrieval architecture while maintaining clean software engineering principles, reproducibility, and modular design.

---

## Objectives

This project aims to demonstrate:

- Recommendation system fundamentals
- Machine learning system design
- Production-oriented code organization
- Model evaluation and experimentation
- API-based model serving
- Reproducible ML workflows
- MLOps best practices

---

## Planned Architecture

```text
                    Offline Training
┌─────────────────────────────────────────────┐
│ User-Item Interaction Data                  │
│ Feature Engineering                         │
│ Model Training                              │
│ Evaluation                                  │
│ Artifact Export                             │
└─────────────────────────────────────────────┘
                     │
                     ▼
           Trained Model Artifacts
                     │
                     ▼
                 FastAPI Service
┌─────────────────────────────────────────────┐
│ Request Validation                          │
│ Recommendation Service                      │
│ Candidate Retrieval                         │
│ Ranking                                     │
│ Top-K Recommendations                       │
└─────────────────────────────────────────────┘
```

---

## Models

### Baseline Models

- Popularity-Based Recommender
- Matrix Factorization

### Neural Models

- Two-Tower Retrieval Model (PyTorch)

Future extensions may include:

- Feature-based ranking models
- Approximate nearest-neighbor retrieval
- Hybrid recommendation architectures

---

## Tech Stack

### Machine Learning

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-Learn

### Backend

- FastAPI
- Pydantic

### MLOps

- MLflow
- Docker

### Testing

- Pytest

### Retrieval (Planned)

- FAISS

---

## Current Progress

### Repository Setup

- [x] Project structure
- [x] GitHub repository
- [x] Python virtual environment
- [x] Testing framework setup

### Recommendation Models

- [ ] Popularity Recommender
- [ ] Matrix Factorization
- [ ] Two-Tower Neural Recommender

### Evaluation

- [ ] Recall@K
- [ ] Precision@K
- [ ] NDCG@K

### API

- [ ] FastAPI service
- [ ] Recommendation endpoint
- [ ] Health endpoint

### MLOps

- [ ] MLflow experiment tracking
- [ ] Docker deployment
- [ ] CI/CD pipeline

---

## Project Structure

```text
recsysops/
│
├── app/
│   ├── api/
│   ├── core/
│   ├── schemas/
│   └── services/
│
├── training/
├── inference/
├── configs/
├── artifacts/
├── tests/
├── scripts/
├── docker/
│
├── README.md
├── .gitignore
└── pyproject.toml
```

---

## Learning Goals

This project is intentionally being developed from first principles.

Rather than relying heavily on high-level recommendation frameworks, the focus is on understanding:

- Why recommendation systems work
- How embeddings are learned
- Offline vs online serving architectures
- Retrieval and ranking systems
- Model evaluation methodology
- Production deployment workflows

---

## Future Work

- Two-tower retrieval architecture
- Candidate generation pipeline
- Vector similarity search
- Experiment tracking with MLflow
- Containerized deployment
- Monitoring and observability
- Recommendation explainability

---

## Author

**Farshad Haddadi**

Computer Science Student  
University of Toronto

Interested in:

- Machine Learning Engineering
- Recommendation Systems
- Applied AI
- MLOps
- Distributed Systems

---

## Disclaimer

This repository is a learning-focused engineering project. The goal is not only to build a recommendation system but to develop a deep understanding of the software engineering, machine learning, and deployment principles required to operate such systems in production.
