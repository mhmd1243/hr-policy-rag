import warnings
warnings.filterwarnings('ignore')

# Core imports
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_engine import RAGEngine, PDF_PATH, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, DEFAULT_TOP_K


rag_engine = RAGEngine(pdf_path=PDF_PATH)


# Load the PDF + models once when the server starts
@asynccontextmanager
async def lifespan(app):
    start_time = time.time()
    rag_engine.build()
    print(f"✅ RAG engine loaded in {time.time() - start_time:.1f}s")
    yield


app = FastAPI(
    title="HR Policy RAG API",
    description="Ask questions about The OpenGov Foundation Policy Manual (Retrieval-Augmented Generation).",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user's question")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=10, description="How many chunks to retrieve")


class SourceChunk(BaseModel):
    page: int
    text: str


class RetrieveResponse(BaseModel):
    question: str
    sources: list[SourceChunk]


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    seconds: float


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "HR Policy RAG API is running. Open /docs to try it."}


@app.get("/health")
def health():
    return {
        "status": "ready" if rag_engine.is_ready else "loading",
        "chunks": len(rag_engine.splits),
    }


@app.get("/info")
def info():
    return {
        "document": PDF_PATH,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
        "chunks": len(rag_engine.splits),
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: QuestionRequest):
    if not rag_engine.is_ready:
        raise HTTPException(status_code=503, detail="The model is still loading, try again in a moment.")

    sources = rag_engine.retrieve(
        request.question,
        top_k=request.top_k,
    )
    return {
        "question": request.question,
        "sources": sources,
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if not rag_engine.is_ready:
        raise HTTPException(status_code=503, detail="The model is still loading, try again in a moment.")

    start_time = time.time()
    try:
        result = rag_engine.ask(
            request.question,
            top_k=request.top_k,
        )
    except Exception as error:
        print(f"❌ /ask failed: {type(error).__name__}: {error}")
        raise HTTPException(status_code=502, detail=f"{type(error).__name__}: {error}")
    result["seconds"] = round(time.time() - start_time, 2)
    return result


# Run from the terminal:  uvicorn api:app --host 0.0.0.0 --port 8000
