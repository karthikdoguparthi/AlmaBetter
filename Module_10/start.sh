#!/bin/bash
echo "Starting CSAT Predictor AI..."
source venv/bin/activate
streamlit run streamlit_app.py --server.port 8000
