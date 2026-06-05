# RecSysOps: End-to-End MLOps Recommendation System

Production-ready recommendation system demonstrating machine learning engineering, MLOps, containerization, CI/CD, and cloud deployment.

## Overview

RecSysOps is an end-to-end recommendation system built using the MovieLens 100K dataset. The project trains and serves multiple recommendation algorithms through a FastAPI service deployed on AWS with automated testing and CI/CD.

### Models Implemented

* Matrix Factorization
* Two-Tower Neural Recommendation Model
* Popularity Baseline

### Features

* FastAPI inference service
* Docker containerization
* GitHub Actions CI/CD
* AWS EC2 deployment
* Experiment tracking
* Model comparison endpoint
* Automated testing
* Swagger API documentation

## Screenshots

### API Documentation

![Swagger UI](docs/images/swagger-ui.png)

### Recommendation Endpoint

![Recommendation Endpoint](docs/images/recommend-endpoint.png)

### Model Comparison

![Model Comparison](docs/images/compare-models.png)

### CI/CD Pipeline

![GitHub Actions](docs/images/github-actions.png)

### System Architecture

![Architecture](docs/images/architecture-diagram.png)

## API Endpoints

### Health Check

GET /health

Response:

{
"status": "ok"
}

### Recommendations

POST /recommend

Request:

{
"user_id": 1,
"k": 10,
"model_name": "matrix_factorization"
}

### Metrics

GET /metrics

Returns training metrics and experiment information.

### Model Comparison

GET /compare-models

Compares available recommendation models and identifies the best-performing model.

## Tech Stack

### Machine Learning

* PyTorch
* NumPy
* Pandas
* Scikit-Learn

### MLOps

* Docker
* GitHub Actions
* AWS EC2

### Serving

* FastAPI
* Uvicorn

## Testing

Run locally:

pytest

## Local Development

git clone <repo-url>

docker build -t recsysops .

docker run -p 8000:8000 recsysops

Open:

http://localhost:8000/docs

## Live Demo

Swagger UI:

http://18.116.199.104:8000/docs

Health Check:

http://18.116.199.104:8000/health

## Example Results

Current deployed models:

* Matrix Factorization
* Two Tower
* Popularity Baseline

Model comparison endpoint automatically evaluates available experiment artifacts and identifies the best-performing model.

## Future Improvements

* Feature Store integration
* MLflow experiment tracking
* Kubernetes deployment
* Continuous training pipeline
* Monitoring and observability

## Author

Farshad Haddadi

GitHub:
https://github.com/farshad-haddadi
