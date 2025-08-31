# --- run in terminal ---
# streamlit run app.py

import sys

try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass


import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# ---------------------------
# 0) ENV & CONSTANTS
# ---------------------------
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] 
# GROQ_API_KEY = os.getenv("GROQ_API_KEY") == "GROQ_API_KEY"
# assert GROQ_API_KEY, "Set GROQ_API_KEY in your environment or .env file"

CSV_PATH = "Library_data.csv"
PERSIST_DIR = "chroma_library"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------
# 1) HELPER: LOAD DOCUMENTS
# ---------------------------
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

    return [row_to_document(r) for _, r in df.iterrows()]

# ---------------------------
# 2) BUILD / LOAD VECTORSTORE
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_vectorstore(_docs):  # <-- leading underscore fixes caching
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.isdir(PERSIST_DIR) and any(
        name in set(os.listdir(PERSIST_DIR)) for name in ["chroma.sqlite3", "index", "index.pkl"]
    ):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    else:
        vs = Chroma.from_documents(_docs, embedding, persist_directory=PERSIST_DIR)
        vs.persist()
        return vs

# ---------------------------
# 3) BUILD QA CHAIN
# ---------------------------
@st.cache_resource(show_spinner=False)
def build_qa(_retriever):
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
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template.strip()
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
    )

# ---------------------------
# 4) STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="📚 Library Assistant", layout="centered")
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>📚 Library Assistant MMEC</h1>", unsafe_allow_html=True)
st.write("Ask me about books in your library database of MMEC.")

# Load docs & vectorstore
docs = load_csv_to_docs(CSV_PATH)
vectorstore = load_vectorstore(docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa = build_qa(retriever)

# Input box
query = st.text_input("Enter your question:", placeholder="e.g. Which book is beginner friendly for physics?")
if query:
    with st.spinner("Searching library..."):
        result = qa({"query": query})

    st.subheader("Answer")
    st.text(result["result"])

    with st.expander("📌 Sources"):
        for i, doc in enumerate(result.get("source_documents", []), 1):
            st.markdown(f"**Source {i}:**")
            st.text(doc.page_content)
