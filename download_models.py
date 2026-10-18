import warnings
warnings.filterwarnings('ignore')

# Core imports
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from rag_engine import EMBEDDING_MODEL_NAME, MODELS_CACHE_DIR

# Download the FastEmbed model once (during the Docker build),
# so the Space does not download it on every start.
embedding = FastEmbedEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    cache_dir=MODELS_CACHE_DIR,
)
embedding.embed_query("warm up")
print(f"✅ Downloaded {EMBEDDING_MODEL_NAME} to {MODELS_CACHE_DIR}/")
