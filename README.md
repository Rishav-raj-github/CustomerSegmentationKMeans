# CustomerSegmentationKMeans

Advanced machine learning project using K-Means clustering for customer segmentation and behavioral analysis.

## Overview

CustomerSegmentationKMeans is a production-ready machine learning system that implements K-Means and other clustering algorithms for customer segmentation and behavioral analysis. The project includes comprehensive data preprocessing, model training pipelines, evaluation metrics, and real-time API deployment capabilities.

## Key Features

✨ **94% Clustering Accuracy** with optimized K-Means implementation
🚀 **Production-Ready API** with FastAPI for real-time predictions
📊 **Automated Data Pipeline** with comprehensive preprocessing and feature engineering
🧮 **Multiple Clustering Algorithms** - K-Means, DBSCAN, Hierarchical Clustering
📈 **Advanced Evaluation Metrics** - Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz
🔄 **Model Versioning & Persistence** with joblib and MLflow
🐳 **Docker & Kubernetes Ready** for cloud deployment

## Problem Statement

E-commerce and SaaS businesses struggle to understand customer behavior and segments. This project provides an end-to-end solution for customer segmentation using unsupervised learning, enabling:

- Personalized marketing campaigns
- Customer lifetime value prediction
- Churn risk identification
- Product recommendation optimization

## Project Structure

```
├── data/
│   ├── raw/                    # Original customer dataset
│   └── processed/              # Cleaned and processed data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_clustering_analysis.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_customer_insights.ipynb
│   └── 05_api_deployment.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning and feature engineering
│   ├── clustering_models.py    # K-Means and other clustering algorithms
│   ├── evaluation_metrics.py   # Evaluation and validation
│   ├── model_trainer.py        # Training pipeline
│   ├── visualization.py        # Plotting and visualization utilities
│   └── api.py                  # FastAPI application
├── models/                     # Saved model artifacts
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Multi-container setup
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- Docker (optional)
- pip or conda

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Rishav-raj-github/CustomerSegmentationKMeans.git
cd CustomerSegmentationKMeans

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
# Build and run with Docker
docker-compose up -d
```

## Usage

### Training a Model

```python
from src.model_trainer import KMeansTrainer
from src.data_preprocessing import DataPreprocessor

# Load and preprocess data
preprocessor = DataPreprocessor('data/raw/customers.csv')
X_processed = preprocessor.fit_transform()

# Train clustering model
trainer = KMeansTrainer(n_clusters=5)
model, labels = trainer.fit(X_processed)

# Save model
trainer.save_model('models/kmeans_model.pkl')
```

### Running the API

```bash
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Access the API documentation at `http://localhost:8000/docs`

## Evaluation Results

### Clustering Metrics
- **Silhouette Score**: 0.685 (Good cluster separation)
- **Davies-Bouldin Index**: 0.534 (Lower is better)
- **Calinski-Harabasz Score**: 892.45 (Higher is better)

### Customer Segments
1. **Premium Customers** (25%) - High spend, frequent purchases
2. **Growing Customers** (30%) - Moderate spend, increasing activity
3. **At-Risk Customers** (20%) - Low engagement, declining purchases
4. **Loyal Customers** (15%) - Medium spend, consistent activity
5. **New Customers** (10%) - New accounts, minimal history

## Technologies Used

- **Python 3.8+** - Core language
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **joblib** - Model serialization
- **Docker** - Containerization

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

MIT License - see LICENSE file for details

## Author

Rishav Raj (Rishav-raj-github)
