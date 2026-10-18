import warnings
warnings.filterwarnings('ignore')

# Core imports
import os
import requests
import streamlit as st


# Settings
API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180

SAMPLE_QUESTIONS = [
    "How many vacation days should employees take per year?",
    "What percentage of the health insurance premium does the company pay?",
    "Which expenses need Board approval?",
    "When is payroll paid?",
    "How do I file a reimbursement request?",
]


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide",
)

# ---------------------------------------------------------------
# Helpers that talk to the FastAPI backend
# ---------------------------------------------------------------
def get_api_status():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except Exception:
        return {"status": "offline", "chunks": 0}


def get_api_info():
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        return response.json()
    except Exception:
        return {}


def ask_api(question, top_k):
    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": question,
            "top_k": top_k,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def show_sources(sources):
    with st.expander(f"📄 Sources ({len(sources)} chunks)"):
        for source in sources:
            st.markdown(f"**Page {source['page']}**")
            st.caption(source["text"])


# ---------------------------------------------------------------
# API status box (checks again by itself until the API is ready)
# ---------------------------------------------------------------
status = get_api_status()
api_is_ready = status["status"] == "ready"

if api_is_ready:
    check_every = None      # ready -> stop checking
else:
    check_every = 3         # not ready -> check every 3 seconds


@st.fragment(run_every=check_every)
def api_status_box():
    current_status = get_api_status()

    if current_status["status"] == "ready":
        if not api_is_ready:
            # The API just came up -> reload the whole page once (models info, etc.)
            st.rerun()
        st.success(f"✅ API ready · {current_status['chunks']} chunks indexed")
    elif current_status["status"] == "loading":
        st.warning("⏳ API is loading the models…")
    else:
        st.error(f"❌ API not reachable at {API_URL} — checking again every {check_every}s…")


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_status_box()

    top_k = st.slider(
        "Chunks to retrieve (top_k)",
        min_value=1,
        max_value=8,
        value=4,
    )

    info = get_api_info()
    if len(info) > 0:
        st.markdown("**Models**")
        st.caption(f"Embeddings: `{info['embedding_model']}`")
        st.caption(f"LLM (Groq): `{info['llm_model']}`")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------
# Main page
# ---------------------------------------------------------------
st.title("📘 HR Policy Assistant")
st.write("Ask anything about **The OpenGov Foundation Policy Manual**. "
         "Answers are generated only from the manual, with the pages they came from.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sample question buttons
st.markdown("**Try a question:**")
sample_columns = st.columns(len(SAMPLE_QUESTIONS))
clicked_question = None
for index in range(len(SAMPLE_QUESTIONS)):
    with sample_columns[index]:
        if st.button(SAMPLE_QUESTIONS[index], key=f"sample_{index}", use_container_width=True):
            clicked_question = SAMPLE_QUESTIONS[index]

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            show_sources(message["sources"])

# New question (typed or clicked)
typed_question = st.chat_input("Type your question here…")
question = typed_question if typed_question else clicked_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the manual and writing the answer…"):
            try:
                result = ask_api(question, top_k)
                st.markdown(result["answer"])
                st.caption(f"⏱️ {result['seconds']}s")
                show_sources(result["sources"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except requests.exceptions.HTTPError as error:
                try:
                    detail = error.response.json().get("detail", str(error))
                except ValueError:
                    detail = error.response.text
                st.error(f"API error ({error.response.status_code}): {detail}")
            except Exception as error:
                st.error(f"Could not reach the API: {error}")


# Run from the terminal:  streamlit run streamlit_app.py
