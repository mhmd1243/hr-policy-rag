import warnings
warnings.filterwarnings('ignore')

# Core imports
import os
from huggingface_hub import HfApi

# Settings
# Before running:  export HF_TOKEN=hf_xxx   (a token with "write" access from huggingface.co/settings/tokens)
HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_ID = os.getenv("SPACE_ID", "your-username/hr-policy-rag")   # change your-username

if HF_TOKEN is None:
    raise SystemExit("❌ Set HF_TOKEN first:  export HF_TOKEN=hf_xxx")

api = HfApi(token=HF_TOKEN)

# 1) Create the Space (Docker SDK) if it does not exist yet
api.create_repo(
    repo_id=SPACE_ID,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
)
print(f"✅ Space ready: {SPACE_ID}")

# 2) Upload the whole project folder (handles the PDF as a binary file for you)
api.upload_folder(
    folder_path=".",
    repo_id=SPACE_ID,
    repo_type="space",
    ignore_patterns=[
        "__pycache__/*",
        "*.pyc",
        ".venv/*",
        "venv/*",
        ".git/*",
        ".env",
        "models_cache/*",
        "*.ipynb",
    ],
    commit_message="Deploy HR Policy RAG app",
)
print(f"✅ Uploaded. The Space is building now: https://huggingface.co/spaces/{SPACE_ID}")
