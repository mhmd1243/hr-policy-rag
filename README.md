---
title: HR Policy RAG
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 📘 HR Policy RAG Assistant

تطبيق RAG للإجابة عن الأسئلة من دليل سياسات الموظفين (`data/HR_Policy.pdf`).

- **RAG:** LangChain + FastEmbed + Chroma + GPT-OSS 120B (Groq)
- **Backend:** FastAPI (`api.py`)
- **Frontend:** Streamlit (`streamlit_app.py`)
- **Deployment:** Hugging Face Spaces (Docker)

