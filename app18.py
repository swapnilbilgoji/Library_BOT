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

# 🔗 CHANGE THIS to your deployed backend URL when on cloud
BACKEND_URL = "https://backend-library-folder-production.up.railway.app"

st.set_page_config(page_title="📚 Super Library Assistant", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .stTabs [role="tablist"] {
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>📚 Super Library Assistant - MMEC</h1>",
    unsafe_allow_html=True,
)

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
# QA CHAIN (for best-match text)
# ---------------------------
@st.cache_resource(show_spinner=False)
def build_qa(_retriever):
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

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
# LOAD RAG COMPONENTS
# ---------------------------
docs = load_csv_to_docs(CSV_PATH)
vectorstore = load_vectorstore(docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa = build_qa(retriever)

# ---------------------------
# SESSION STATE (MODALS)
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
# TABS (3 only now)
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

            if not top_docs:
                st.info("No relevant books found.")
            else:
                for i, d in enumerate(top_docs, start=1):
                    meta = d.metadata
                    accession = str(meta.get("AccessionNo", "")).strip()
                    title = meta.get("Title", "Unknown Title")
                    author = meta.get("Author", "Unknown Author")
                    subject = meta.get("Subject", "Unknown Subject")
                    edition = meta.get("Edition", "N/A")
                    year = meta.get("Year", "N/A")

                    copies_info = None
                    available_text = "Unknown"
                    rack_text = ""
                    copies_zero = False
                    if accession:
                        try:
                            resp = requests.get(f"{BACKEND_URL}/books/status/{accession}", timeout=5)
                            if resp.status_code == 200:
                                copies_info = resp.json()
                                av = copies_info.get("available_copies", 0)
                                total = copies_info.get("total_copies", 0)
                                available_text = f"{av} / {total}"
                                rack_text = copies_info.get("rack_location", "")
                                if av <= 0:
                                    copies_zero = True
                            else:
                                available_text = "Backend book info not found"
                        except Exception as e:
                            available_text = "Backend not reachable"

                    with st.container(border=True):
                        st.markdown(f"#### {i}) {title}")
                        st.write(f"**Author:** {author}")
                        st.write(f"**Accession No:** {accession}")
                        st.write(f"**Subject:** {subject}")
                        st.write(f"**Edition / Year:** {edition} / {year}")
                        st.write(f"**Copies Available:** {available_text}")
                        if rack_text:
                            st.write(f"**Rack / Shelf Location:** {rack_text}")

                        if copies_zero:
                            st.warning("❌ All copies are currently issued. Please come after 15 days.")
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

    # ISSUE MODAL
    if st.session_state.get("issue_modal_open") and st.session_state.get("selected_accession_issue"):
        with st.modal("Issue Book"):
            st.write(f"Book Accession No: {st.session_state['selected_accession_issue']}")
            usn_issue = st.text_input("Enter your USN:", key="modal_usn_issue")

            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.button("Confirm Issue"):
                    if not usn_issue:
                        st.error("Please enter your USN.")
                    else:
                        try:
                            resp = requests.post(
                                f"{BACKEND_URL}/books/issue/request",
                                json={"usn": usn_issue, "accession_no": st.session_state["selected_accession_issue"]},
                                timeout=10
                            )
                            if resp.status_code == 200:
                                st.success(resp.json().get("message", "Book issued successfully."))
                                st.session_state["issue_modal_open"] = False
                                st.session_state["selected_accession_issue"] = None
                                st.rerun()
                            else:
                                st.error(f"Error: {resp.text}")
                        except Exception as e:
                            st.error(f"Could not connect to backend: {e}")
            with col_cancel:
                if st.button("Cancel"):
                    st.session_state["issue_modal_open"] = False
                    st.session_state["selected_accession_issue"] = None
                    st.rerun()

    # WISHLIST MODAL
    if st.session_state.get("wishlist_modal_open") and st.session_state.get("selected_accession_wishlist"):
        with st.modal("Add Book to Wishlist"):
            st.write(f"Book Accession No: {st.session_state['selected_accession_wishlist']}")
            usn_wish = st.text_input("Enter your USN:", key="modal_usn_wish")

            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.button("Save to Wishlist"):
                    if not usn_wish:
                        st.error("Please enter your USN.")
                    else:
                        try:
                            resp = requests.post(
                                f"{BACKEND_URL}/wishlist/add",
                                json={"usn": usn_wish, "accession_no": st.session_state["selected_accession_wishlist"]},
                                timeout=10
                            )
                            if resp.status_code == 200:
                                st.success(resp.json().get("message", "Book added to wishlist."))
                                st.session_state["wishlist_modal_open"] = False
                                st.session_state["selected_accession_wishlist"] = None
                                st.rerun()
                            else:
                                st.error(f"Error: {resp.text}")
                        except Exception as e:
                            st.error(f"Could not connect to backend: {e}")
            with col_cancel:
                if st.button("Cancel", key="cancel_wish"):
                    st.session_state["wishlist_modal_open"] = False
                    st.session_state["selected_accession_wishlist"] = None
                    st.rerun()

# ---------------------------
# TAB 2: ATTENDANCE
# ---------------------------
with tab2:
    st.subheader("Student Library Attendance")

    usn_att = st.text_input("Enter your USN to mark today's attendance:")

    if st.button("Mark Attendance"):
        if not usn_att:
            st.error("Please enter USN.")
        else:
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/attendance/mark",
                    json={"usn": usn_att},
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(data.get("message", "Attendance marked successfully."))
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

# ---------------------------
# TAB 3: MY LIBRARY (VIEW + RENEW + RETURN + WISHLIST)
# ---------------------------
with tab3:
    st.subheader("👤 My Library – Student Section")

    st.write("Login with your USN to see your books, renew, request return, and view wishlist.")
    usn_profile = st.text_input("Enter your USN (works like password):", type="password")

    if st.button("View My Library"):
        if not usn_profile:
            st.error("Please enter your USN.")
        else:
            try:
                resp = requests.get(f"{BACKEND_URL}/student/profile/{usn_profile}", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    student = data.get("student", {})
                    active_books = data.get("active_books", [])
                    wishlist = data.get("wishlist", [])

                    st.markdown("### 🧑‍🎓 Student Details")
                    st.write(f"**Name:** {student.get('name')}")
                    st.write(f"**USN:** {student.get('usn')}")
                    st.write(f"**Branch / Sem:** {student.get('branch')} / {student.get('semester')}")
                    st.write(f"**Phone:** {student.get('phone')}")
                    st.write(f"**Email:** {student.get('email')}")

                    st.markdown("---")
                    st.markdown("### 📚 Active Issued Books")
                    if not active_books:
                        st.info("No active issued books.")
                    else:
                        for b in active_books:
                            with st.container(border=True):
                                st.write(f"**Title:** {b['title']}")
                                st.write(f"**Accession No:** {b['accession_no']}")
                                st.write(f"**Issue Date:** {b['issue_date']}")
                                st.write(f"**Due Date:** {b['due_date']}")
                                st.write(f"**Status:** {b['status']}")

                                if b["status"] in ["issued", "renewed"]:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button(
                                            "🔁 Renew this Book",
                                            key=f"renew_profile_{b['accession_no']}"
                                        ):
                                            try:
                                                resp2 = requests.post(
                                                    f"{BACKEND_URL}/books/renew",
                                                    json={
                                                        "usn": usn_profile,
                                                        "accession_no": b["accession_no"],
                                                        "extra_days": 30
                                                    },
                                                    timeout=10
                                                )
                                                if resp2.status_code == 200:
                                                    st.success("Renewal processed successfully.")
                                                else:
                                                    st.error(f"Error: {resp2.text}")
                                            except Exception as e:
                                                st.error(f"Could not connect to backend: {e}")
                                    with col2:
                                        if st.button(
                                            "📤 Request Return",
                                            key=f"req_ret_profile_{b['accession_no']}"
                                        ):
                                            try:
                                                resp3 = requests.post(
                                                    f"{BACKEND_URL}/books/return/request",
                                                    json={"usn": usn_profile, "accession_no": b["accession_no"]},
                                                    timeout=10
                                                )
                                                if resp3.status_code == 200:
                                                    st.success("Return request sent to librarian.")
                                                else:
                                                    st.error(f"Error: {resp3.text}")
                                            except Exception as e:
                                                st.error(f"Could not connect to backend: {e}")

                    st.markdown("---")
                    st.markdown("### 🔖 Wishlist")
                    if not wishlist:
                        st.info("No books in wishlist.")
                    else:
                        for w in wishlist:
                            with st.container(border=True):
                                st.write(f"**Title:** {w['title']}")
                                st.write(f"**Accession No:** {w['accession_no']}")
                                st.write(f"**Added On:** {w['added_on']}")
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
