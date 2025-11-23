"""
Single-file Streamlit app for MMEC Library Assistant.
- Recreates Chroma DB safely when FORCE_REINDEX=1 environment variable is set (or when reindex button clicked in UI).
- Uses langchain_community HuggingFaceEmbeddings + Chroma vectorstore (kept consistent).
- Provides diagnostics and clear error handling for Chroma InternalError; offers to reindex.

Run:
    FORCE_REINDEX=1 streamlit run library_app_single_file.py
or just:
    streamlit run library_app_single_file.py

Make sure your venv has required packages installed (streamlit, langchain_community, chromadb, langchain_groq, python-dotenv, pandas).
Put your CSV as `Library_data.csv` in the same folder and add GROQ_API_KEY to Streamlit secrets or env.
"""

import os
import sys
import time
import shutil
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Use only the community HuggingFace embeddings and Chroma vectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Optional: chromadb errors class for catching rust internal errors
try:
    import chromadb
    from chromadb.errors import InternalError as ChromaInternalError
except Exception:
    chromadb = None
    ChromaInternalError = Exception

# ---------------------------
# CONFIG / CONSTANTS
# ---------------------------
load_dotenv()

CSV_PATH = os.environ.get("CSV_PATH", "Library_data.csv")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "chroma_library")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FORCE_REINDEX = os.environ.get("FORCE_REINDEX", "0") == "1"

# Get GROQ API key (streamlit secrets preferred)
GROQ_API_KEY = None
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ---------------------------
# HELPERS
# ---------------------------

def backup_persist_dir(dirpath: str) -> str:
    """Move existing persist dir to a timestamped backup and return backup path."""
    if not os.path.isdir(dirpath):
        return ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{dirpath}_backup_{ts}"
    shutil.move(dirpath, backup)
    return backup


def safe_remove_persist_dir(dirpath: str):
    if not os.path.isdir(dirpath):
        return
    try:
        shutil.rmtree(dirpath)
    except Exception:
        # fallback to move
        backup_persist_dir(dirpath)


@st.cache_data(show_spinner=False)
def load_csv_to_docs(path: str):
    df = pd.read_csv(path)

    col_map = {
        "Acc. No.": "AccessionNo",
        "Title": "Title",
        "Author": "Author",
        "Place": "Place",
        "Publisher": "Publisher",
        "Year": "Year",
        "Edition": "Edition",
        "Language": "Language",
        "Subject": "Subject",
    }
    df = df.rename(columns=col_map).fillna("")

    def row_to_document(row):
        content = (
            f"The Accession no of book is: {row['AccessionNo']}\n"
            f"The Title is: {row['Title']}\n"
            f"The Author is: {row['Author']}\n"
            f"The Edition is: {row['Edition']}\n"
            f"The Language is: {row['Language']}\n"
            f"The Subject is: {row['Subject']}\n"
            f"The Year of Publishing: {row['Year']}"
        )
        return Document(page_content=content, metadata=row.to_dict())

    docs = [row_to_document(r) for _, r in df.iterrows()]
    return docs


@st.cache_resource(show_spinner=False)
def build_embedding(model_name: str = EMBED_MODEL):
    emb = HuggingFaceEmbeddings(model_name=model_name)
    return emb


@st.cache_resource(show_spinner=False)
def load_vectorstore(docs, persist_dir: str, embedding):
    """Load or create Chroma vectorstore. Returns the vectorstore instance."""
    # If persistence exists, load it with embedding_function that will be used for queries
    if os.path.isdir(persist_dir) and any(name in set(os.listdir(persist_dir)) for name in ["chroma.sqlite3", "index", "index.pkl"]):
        vs = Chroma(persist_directory=persist_dir, embedding_function=embedding.embed_query)
        return vs
    else:
        # Create a new collection from docs and persist
        vs = Chroma.from_documents(documents=docs, embedding_function=embedding.embed_documents, persist_directory=persist_dir)
        vs.persist()
        return vs


def build_qa_chain(retriever):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Put it in Streamlit secrets or environment variable GROQ_API_KEY.")

    llm = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)

    template = """
You are a helpful library assistant.
Answer ONLY using the provided context. If a requested field is missing in the context, write "Information not available" for that field.
If no relevant context is retrieved at all, reply exactly with: "Information not available." 

Your answer must be in EXACTLY this format:

The Accession no of book is: <accession_no>
The Title is: <title>
The Author is: <author>
The Edition is: <edition>
The Language is: <language>
The Subject is: <subject>
The Year of Publishing: <year>

If multiple books are in context, choose the single best match and output ONLY ONE set.

Context:
{context}

Question:
{question}
"""

    custom_prompt = PromptTemplate(input_variables=["context", "question"], template=template.strip())

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
    )


# ---------------------------
# STREAMLIT UI / STARTUP
# ---------------------------

st.set_page_config(page_title="📚 Library Assistant", layout="centered")
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>📚 Library Assistant MMEC</h1>", unsafe_allow_html=True)
st.write("Ask me about books in your library database of MMEC.")

# UI: allow user to force reindex from the page
force_reindex_ui = st.sidebar.button("Force reindex (recreate Chroma DB)")

# also support env var
if FORCE_REINDEX:
    st.sidebar.write("FORCE_REINDEX environment variable is set -> will reindex on startup.")

# show basic diagnostics
st.sidebar.write(f"CSV: {CSV_PATH}")
st.sidebar.write(f"Persist dir: {PERSIST_DIR}")
st.sidebar.write(f"Embedding model: {EMBED_MODEL}")

# Load documents
if not os.path.exists(CSV_PATH):
    st.error(f"CSV file not found: {CSV_PATH}. Place your Library_data.csv in the app folder.")
    st.stop()

with st.spinner("Loading CSV..."):
    docs = load_csv_to_docs(CSV_PATH)

# Decide if we should reindex
do_reindex = FORCE_REINDEX or force_reindex_ui
if do_reindex:
    st.sidebar.warning("Reindex requested: backing up and recreating the persist directory...")
    backup = backup_persist_dir(PERSIST_DIR)
    if backup:
        st.sidebar.write(f"Persist directory backed up to: {backup}")

# Build embeddings and vectorstore, catch common chroma failures
embedding = build_embedding(EMBED_MODEL)

try:
    with st.spinner("Loading/creating vectorstore (this may take some time)..."):
        if do_reindex:
            safe_remove_persist_dir(PERSIST_DIR)
        vectorstore = load_vectorstore(docs, PERSIST_DIR, embedding)
except ChromaInternalError as e:
    st.error("Chroma returned an internal error while loading the vectorstore.\nSee details in terminal where Streamlit was started.")
    st.caption("You can try: setting FORCE_REINDEX=1 and restarting, or delete the persist directory to re-create it.")
    st.write("Exception:")
    st.text(str(e))
    st.stop()
except Exception as e:
    st.error("Error while creating/loading the vectorstore. Check stdout/traceback in the terminal.")
    st.exception(e)
    st.stop()

# show some diagnostics
try:
    col_count = None
    try:
        col_count = vectorstore._collection.count()
    except Exception:
        # safe fallback using retriever (some wrappers differ)
        try:
            col_count = len(vectorstore._collection.get("ids"))
        except Exception:
            col_count = "unknown"
    st.sidebar.write(f"Collection count: {col_count}")
    # sample embedding dim
    try:
        sample_emb = embedding.embed_query("test")
        st.sidebar.write(f"Embedding dim (sample): {len(sample_emb)}")
    except Exception as ie:
        st.sidebar.write(f"Could not compute sample embedding: {ie}")
except Exception:
    pass

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

try:
    qa = build_qa_chain(retriever)
except Exception as e:
    st.error("Failed to build the QA chain (likely missing GROQ API key or LLM issue).")
    st.exception(e)
    st.stop()

# Input box
query = st.text_input("Enter your question:", placeholder="e.g. Which book is beginner friendly for physics?")
if query:
    try:
        with st.spinner("Searching library..."):
            result = qa({"query": query})

        st.subheader("Answer")
        st.text(result["result"])

        with st.expander("📌 Sources"):
            for i, doc in enumerate(result.get("source_documents", []), 1):
                st.markdown(f"**Source {i}:**")
                st.text(doc.page_content)
    except ChromaInternalError as e:
        st.error("Chroma InternalError while querying the index. Consider reindexing. See terminal for details.")
        st.exception(e)
        if st.button("Reindex now (safe)"):
            backup = backup_persist_dir(PERSIST_DIR)
            safe_remove_persist_dir(PERSIST_DIR)
            st.experimental_rerun()
    except Exception as e:
        st.error("An error occurred while answering. See details below.")
        st.exception(e)
        st.write("Traceback:")
        st.text(traceback.format_exc())

# Footer note
st.markdown("---")
st.caption("If you still see internal chromadb errors, run this app from a terminal (not on Streamlit Cloud) and paste the full terminal traceback here. For quick recovery, set environment variable FORCE_REINDEX=1 before starting to recreate the index.")
