<div align="center">
  
# 🎯 CSAT Predictor AI
  
**Real-time Customer Satisfaction Scoring using Deep Learning & XGBoost**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, enterprise-grade machine learning application designed to predict and optimize Customer Satisfaction (CSAT) scores for e-commerce interactions. 

[Explore Features](#-features) • [Installation](#-installation--setup) • [Usage](#-running-the-application) • [Architecture](#-repository-structure)

</div>

---

## 🌟 Overview

In the fast-paced e-commerce environment, understanding customer satisfaction *before* they leave a review is critical. 

Built with a highly optimized **XGBoost Classifier** backend and a premium **Streamlit** dashboard frontend, this tool empowers customer service teams, managers, and executives to dynamically predict customer satisfaction scores based on interaction metadata, shift timings, and historical agent performance.

<br>

## 🚀 Features

* ⚡ **Real-time Inference Engine**: Instantly predicts the CSAT score (scaled out of 5.0) and categorizes the interaction as **Satisfied (⭐)**, **Neutral (😐)**, or **Dissatisfied (😞)**.
* 🧠 **Dynamic Feature Engineering**: Agent, Manager, and Supervisor dropdowns are dynamically loaded and mean-encoded behind the scenes, reflecting true historical performance distributions.
* 🏆 **Optimal Strategy Calculator**: A dedicated computational tab that determines the exact combination of categorical and continuous variables required to achieve a flawless 5.0 CSAT score.
* 🎨 **Premium UI/UX**: Built with an immersive dark-mode aesthetic utilizing CSS glassmorphism, animated metrics, and a zero-scroll responsive layout.

<br>

## 📁 Repository Structure

```text
Module_110/
├── streamlit_app.py          # Main Streamlit frontend and prediction routing
├── start.sh                  # One-click shell script to launch the local server
├── csat_prediction_model.pkl # Trained ML Model (XGBoost)
├── csat_scaler.pkl           # Preprocessing StandardScaler artifact
├── csat_mean_mappings.pkl    # Target mean encodings mapping dictionary
├── csat_label_encoders.pkl   # Serialized LabelEncoders for categorical features
├── requirements.txt          # Explicit Python dependency list
└── .gitignore                # Git exclusion rules
```

<br>

## ⚙️ Installation & Setup

To run this project locally, ensure you have **Python 3.9+** installed on your system.

**1. Clone the repository:**
```bash
git clone https://github.com/karthikdoguparthi/AlmaBetter.git
cd AlmaBetter/Module_110
```

**2. Create an isolated virtual environment:**
```bash
# For macOS and Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

<br>

## 🎮 Running the Application

This project includes a built-in startup script for seamless deployment on UNIX-based systems.

**For macOS / Linux:**
```bash
# Make the startup script executable
chmod +x start.sh

# Launch the server
./start.sh
```

**For Windows (or manual execution):**
```bash
streamlit run streamlit_app.py --server.port 8000
```

Once the server initializes, open your web browser and navigate to:  
👉 **http://localhost:8000**
