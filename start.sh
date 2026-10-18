#!/bin/bash
# Start the FastAPI backend in the background (internal port 8000)
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend on the public port 7860 (the port Hugging Face Spaces exposes)
streamlit run streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
