# 🎯 CSAT Predictor AI

A robust, real-time Machine Learning application that predicts Customer Satisfaction (CSAT) scores for e-commerce interactions. 

Built with an **XGBoost Classifier** in the backend and a modern, glassmorphism-styled **Streamlit** dashboard in the frontend, this tool empowers customer service teams to anticipate customer satisfaction based on interaction metadata.

## 🚀 Features

- **Real-time Prediction**: Instantly predicts the CSAT score (out of 5.0) and determines whether the interaction results in a Satisfied, Neutral, or Dissatisfied customer.
- **Dynamic Feature Mapping**: Dropdowns for Agents, Managers, and Supervisors are dynamically loaded and mean-encoded behind the scenes to accurately reflect individual performance histories.
- **Optimal Strategy Engine**: A dedicated tab that calculates and displays the exact optimal parameters mathematically required to achieve a flawless 5.0 CSAT score.
- **Premium UI**: Custom CSS including glassmorphism panels, gradients, and animated components for a seamless user experience without vertical scrolling.

## 📁 Repository Structure

```
.
├── streamlit_app.py          # Main Streamlit frontend and prediction logic
├── start.sh                  # Shell script to easily launch the application
├── csat_prediction_model.pkl # Trained XGBoost Model
├── csat_scaler.pkl           # StandardScaler artifact
├── csat_mean_mappings.pkl    # Target mean encodings for agents/managers
├── csat_label_encoders.pkl   # LabelEncoders for categorical features
├── requirements.txt          # Python dependency list
└── .gitignore
```

*(Note: The `venv/` folder and raw data files are deliberately excluded to keep the repository lightweight.)*

## ⚙️ Installation & Setup

To run this project locally, ensure you have **Python 3.9+** installed. 

**1. Clone the repository and navigate to the folder:**
```bash
git clone <your-repo-link>
cd Module_110
```

**2. Create and activate a Python virtual environment:**
```bash
# On Mac/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎮 Running the Application

For macOS and Linux users, you can use the provided bash script to start the server:

```bash
# Ensure the script is executable
chmod +x start.sh

# Run the app
./start.sh
```

Alternatively, or if you are on Windows, you can start the application directly using Streamlit:

```bash
streamlit run streamlit_app.py --server.port 8000
```

Once running, open your web browser and navigate to:
**http://localhost:8000**
