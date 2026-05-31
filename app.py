import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ── PDF processing ────────────────────────────────────────────────────────────
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    docs = PyPDFLoader(tmp_path).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )

    os.unlink(tmp_path)
    return retriever

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF Chatbot")
    st.caption("RAG · ChromaDB · Mistral")
    st.divider()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Processing PDF..."):
            st.session_state.retriever = process_pdf(uploaded_file)
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_history = []
        st.success(f"✅ Ready: {uploaded_file.name}")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("Chat with your PDF")

if not st.session_state.retriever:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    user_input = st.chat_input("Ask anything about your PDF...")

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.retriever.invoke(user_input)
                context = "\n\n".join([doc.page_content for doc in docs])

                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document." """),
                    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
                ])

                response = get_llm().invoke(
                    prompt.invoke({"context": context, "question": user_input})
                )

            st.write(response.content)

        st.session_state.chat_history.append({"role": "assistant", "content": response.content})
