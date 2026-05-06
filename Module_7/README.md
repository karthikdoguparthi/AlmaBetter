# 🌍 Travel & Hotel MLOps System

A comprehensive MLOps production pipeline for travel price prediction, gender classification, and personalized hotel recommendations. This project demonstrates end-to-end machine learning lifecycle management—from training and tracking to deployment and orchestration.

## 🚀 Key Features
- **Flight Price Prediction**: Gradient Boosting regression for real-time price estimation.
- **Gender Classification**: Predicts user gender based on travel behavior.
- **Hotel Recommendation**: Collaborative filtering system using Cosine Similarity.
- **REST API**: Production-ready Flask API serving predictions on port 5001.
- **Interactive Dashboard**: Streamlit interface for data visualization and recommendation testing.

## 🛠 Technology Stack
- **Languages**: Python 3.9+
- **Machine Learning**: Scikit-Learn, Pandas, Numpy, Imbalanced-Learn
- **API Framework**: Flask
- **Dashboard**: Streamlit
- **MLOps Tools**:
  - **Containerization**: Docker
  - **Orchestration**: Kubernetes, Apache Airflow
  - **CI/CD**: Jenkins
  - **Tracking**: MLFlow

## 📂 Project Structure
| File/Folder | Purpose |
| :--- | :--- |
| `train_models.py` | Model training and serialization script. |
| `app.py` | Flask REST API for model serving. |
| `streamlit_app.py` | Web dashboard for insights and recommendations. |
| `dags/` | Apache Airflow DAGs for workflow automation. |
| `mlflow_tracking.py` | Experiment tracking and model versioning. |
| `Dockerfile` | Containerization configuration. |
| `deployment.yml` | Kubernetes Deployment & Service manifests. |
| `Jenkinsfile` | CI/CD pipeline automation. |

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd travel-mlops-capstone
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Train Models
Ensure `flights.csv`, `hotels.csv`, and `users.csv` are in the root directory.
```bash
python train_models.py
```

### 4. Run the Application
Start the Flask API:
```bash
python app.py
```
Start the Streamlit Dashboard (in a new terminal):
```bash
streamlit run streamlit_app.py
```

## 🐳 Deployment

### Docker
```bash
docker build -t travel-api .
docker run -p 5001:5001 travel-api
```

### Kubernetes
```bash
kubectl apply -f deployment.yml
```

## 📊 Monitoring & Tracking
View MLFlow experiments:
```bash
python mlflow_tracking.py
mlflow ui
```
Access the UI at `http://localhost:5000`.
