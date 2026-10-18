#!/bin/bash
# Run the whole app locally with one command (Git Bash):  bash run_local.sh
source venv/Scripts/activate

# 1) FastAPI backend in the background
uvicorn api:app --port 8000 &
API_PID=$!

# Stop the backend when Streamlit is closed (Ctrl+C)
trap "kill $API_PID 2>/dev/null" EXIT

# 2) Streamlit frontend
streamlit run streamlit_app.py
