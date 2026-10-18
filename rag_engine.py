import warnings
warnings.filterwarnings('ignore')

# Core imports
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
except ImportError:
    # LangChain 1.x moved the old chains into langchain_classic
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.chains import create_retrieval_chain


# Read GROQ_API_KEY (and any other setting) from the .env file if it exists
load_dotenv()

# Settings (all can be changed with environment variables)
PDF_PATH = os.getenv("PDF_PATH", "data/HR_Policy.pdf")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")   # FastEmbed default model
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODELS_CACHE_DIR = os.getenv("MODELS_CACHE_DIR", "models_cache")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 4

PROMPT_TEMPLATE = """
You are an HR assistant for The OpenGov Foundation.
Use the following context from the company's Policy Manual to answer the question.
If you don't know the answer, just say "I don't know."

Context:
{context}

Question: {input}
"""


class RAGEngine:
    def __init__(self, pdf_path=PDF_PATH):
        self.pdf_path = pdf_path
        self.splits = []
        self.vectorstore = None
        self.llm = None
        self.prompt = None
        self.is_ready = False

    # ✅ Step 1: Load and split the PDF
    def load_and_split(self):
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.splits = splitter.split_documents(docs)
        print(f"✅ Loaded {len(docs)} pages -> {len(self.splits)} chunks")

    # ✅ Step 2: Embed and store in the vector DB
    def build_vectorstore(self):
        embedding = FastEmbedEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            cache_dir=MODELS_CACHE_DIR,
        )
        self.vectorstore = Chroma.from_documents(
            self.splits,
            embedding=embedding,
        )
        print(f"✅ Vector store ready ({EMBEDDING_MODEL_NAME})")

    # ✅ Step 3: Initialize the LLM (Groq API via OpenAI-compatible endpoint)
    def build_llm(self):
        api_key = os.getenv("GROQ_API_KEY")
        if api_key is None:
            raise RuntimeError("❌ GROQ_API_KEY is not set. Add it as an environment variable / Space secret.")

        self.llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            openai_api_base=LLM_BASE_URL,
            openai_api_key=api_key,
            temperature=0.2,
            max_tokens=1024,
            timeout=60,
            max_retries=1,
            extra_body={"reasoning_effort": "low"},   # gpt-oss thinks less -> faster answers
        )
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        print(f"✅ LLM ready ({LLM_MODEL_NAME})")

    def build(self):
        self.load_and_split()
        self.build_vectorstore()
        self.build_llm()
        self.is_ready = True

    # ✅ Step 4: Retrieval chain (built per request so top_k can change)
    def get_retrieval_chain(self, top_k):
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": top_k},
        )
        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain

    def format_sources(self, documents):
        sources = []
        for document in documents:
            sources.append({
                "page": int(document.metadata.get("page", 0)) + 1,   # PyPDFLoader pages start at 0
                "text": document.page_content,
            })
        return sources

    def retrieve(self, question, top_k=DEFAULT_TOP_K):
        documents = self.vectorstore.similarity_search(
            question,
            k=top_k,
        )
        return self.format_sources(documents)

    # ✅ Step 5: Query the document
    def ask(self, question, top_k=DEFAULT_TOP_K):
        retrieval_chain = self.get_retrieval_chain(top_k)
        response = retrieval_chain.invoke({"input": question})
        return {
            "question": question,
            "answer": response["answer"],
            "sources": self.format_sources(response["context"]),
        }


# Quick test from the terminal:  python rag_engine.py
if __name__ == "__main__":
    engine = RAGEngine()
    engine.build()

    test_questions = [
        "What is this document about?",
        "How many vacation days should employees take per year?",
    ]
    for query in test_questions:
        result = engine.ask(query)
        print("\n❓", result["question"])
        print("💬", result["answer"])
        for source in result["sources"]:
            print(f"   📄 page {source['page']}")
