# app.py
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
import requests
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

# ---------------------------
# ENV & CONSTANTS
# ---------------------------
load_dotenv()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

CSV_PATH = "Library_data.csv"
PERSIST_DIR = "chroma_library"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="📚 Super Library Assistant", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f7f9fc; }
    .stTabs [role="tablist"] { justify-content: center; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>📚 Super Library Assistant - MMEC</h1>",
    unsafe_allow_html=True,
)

# ---------------------------
# NEW HELPER (SAFE ADDITION)
# ---------------------------
def get_book_inventory(accession_no: str):
    try:
        resp = requests.get(
            f"{BACKEND_URL}/books/status/{accession_no}",
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

# ---------------------------
# LOAD DOCUMENTS
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
# VECTORSTORE
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_vectorstore(_docs):
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
# QA CHAIN
# ---------------------------
@st.cache_resource(show_spinner=False)
def build_qa(_retriever):
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

    template = """
You are a helpful library assistant.
Answer ONLY using the provided context.

Your answer must be in EXACTLY this format:

The Accession no of book is: <accession_no>
The Title is: <title>
The Author is: <author>
The Edition is: <edition>
The Language is: <language>
The Subject is: <subject>
The Year of Publishing: <year>

Context:
{context}

Question:
{question}
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template.strip()
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

# ---------------------------
# LOAD RAG
# ---------------------------
docs = load_csv_to_docs(CSV_PATH)
vectorstore = load_vectorstore(docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa = build_qa(retriever)

# ---------------------------
# SESSION STATE
# ---------------------------
if "issue_modal_open" not in st.session_state:
    st.session_state["issue_modal_open"] = False
if "wishlist_modal_open" not in st.session_state:
    st.session_state["wishlist_modal_open"] = False
if "selected_accession_issue" not in st.session_state:
    st.session_state["selected_accession_issue"] = None
if "selected_accession_wishlist" not in st.session_state:
    st.session_state["selected_accession_wishlist"] = None

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3 = st.tabs(
    ["🔍 Book Assistant", "✅ Mark Attendance", "👤 My Library"]
)

# ---------------------------
# TAB 1: BOOK ASSISTANT
# ---------------------------
with tab1:
    st.subheader("Ask about books, get recommendations, and issue instantly")

    query = st.text_input(
        "Ask your question:",
        placeholder="e.g. Best book for basic electrical engineering?"
    )

    if st.button("Search Books", type="primary"):
        if not query:
            st.error("Please enter a question.")
        else:
            with st.spinner("Searching library and preparing recommendations..."):
                result = qa({"query": query})
                answer_text = result["result"]
                top_docs = retriever.get_relevant_documents(query)

            st.markdown("### 📌 Best Match (AI Answer)")
            st.text(answer_text)

            st.markdown("---")
            st.markdown("### 📚 Top 5 Recommended Books")

            for i, d in enumerate(top_docs, start=1):
                meta = d.metadata
                accession = str(meta.get("AccessionNo", "")).strip()

                inv = get_book_inventory(accession)
                av = inv.get("available_copies", 0) if inv else 0
                total = inv.get("total_copies", 0) if inv else 0
                rack = inv.get("rack_location", "") if inv else ""

                with st.container(border=True):
                    st.markdown(f"#### {i}) {meta.get('Title','')}")
                    st.write(f"**Author:** {meta.get('Author','')}")
                    st.write(f"**Accession No:** {accession}")
                    st.write(f"**Subject:** {meta.get('Subject','')}")
                    st.write(f"**Edition / Year:** {meta.get('Edition','')} / {meta.get('Year','')}")
                    st.write(f"**Copies Available:** {av} / {total}")
                    if rack:
                        st.write(f"**Rack / Shelf Location:** {rack}")

                    if av <= 0:
                        st.warning("❌ All copies are currently issued.")
                        st.button(
                            "📕 Issue Disabled",
                            disabled=True,
                            key=f"disabled_{accession}_{i}"
                        )
                    else:
                        col_issue, col_wish = st.columns(2)
                        with col_issue:
                            if st.button("📘 Issue this Book", key=f"issue_btn_{accession}_{i}"):
                                st.session_state["issue_modal_open"] = True
                                st.session_state["selected_accession_issue"] = accession
                        with col_wish:
                            if st.button("🔖 Add to Wishlist", key=f"wish_btn_{accession}_{i}"):
                                st.session_state["wishlist_modal_open"] = True
                                st.session_state["selected_accession_wishlist"] = accession

# ---------------------------
# ISSUE MODAL (REAL-TIME REFRESH)
# ---------------------------
if st.session_state.get("issue_modal_open") and st.session_state.get("selected_accession_issue"):
    accession = st.session_state["selected_accession_issue"]
    inv = get_book_inventory(accession)

    with st.modal("Issue Book"):
        st.write(f"**Accession No:** {accession}")

        if inv:
            st.write(f"**Total Copies:** {inv['total_copies']}")
            st.write(f"**Available Copies:** {inv['available_copies']}")
            st.write(f"**Rack:** {inv.get('rack_location','')}")

            if inv["available_copies"] <= 0:
                st.error("❌ No copies available.")
                if st.button("Close"):
                    st.session_state["issue_modal_open"] = False
                    st.session_state["selected_accession_issue"] = None
                    st.rerun()
                st.stop()

        usn_issue = st.text_input("Enter your USN:", key="modal_usn_issue")

        col_ok, col_cancel = st.columns(2)
        with col_ok:
            if st.button("Confirm Issue"):
                resp = requests.post(
                    f"{BACKEND_URL}/books/issue/request",
                    json={"usn": usn_issue, "accession_no": accession},
                    timeout=10
                )
                if resp.status_code == 200:
                    st.success("Book issued successfully.")
                    st.session_state["issue_modal_open"] = False
                    st.session_state["selected_accession_issue"] = None
                    st.rerun()  # 🔥 REAL-TIME REFRESH
                else:
                    st.error(resp.text)

        with col_cancel:
            if st.button("Cancel"):
                st.session_state["issue_modal_open"] = False
                st.session_state["selected_accession_issue"] = None
                st.rerun()

# ---------------------------
# TAB 2: ATTENDANCE
# ---------------------------
with tab2:
    st.subheader("Student Library Attendance")
    usn_att = st.text_input("Enter your USN to mark today's attendance:")

    if st.button("Mark Attendance"):
        resp = requests.post(
            f"{BACKEND_URL}/attendance/mark",
            json={"usn": usn_att},
            timeout=10
        )
        st.success(resp.json().get("message", "Attendance marked."))

# ---------------------------
# TAB 3: MY LIBRARY
# ---------------------------
with tab3:
    st.subheader("👤 My Library – Student Section")
    usn_profile = st.text_input("Enter your USN:", type="password")

    if st.button("View My Library"):
        resp = requests.get(f"{BACKEND_URL}/student/profile/{usn_profile}", timeout=10)
        if resp.status_code == 200:
            st.json(resp.json())
