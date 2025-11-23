# app.py
# Streamlit library assistant — local retrieval (no Chroma / external LLM)
# Usage: streamlit run app.py
import os
import sys
from typing import List, Dict
import pandas as pd
import numpy as np
import streamlit as st

# Attempt several embedding backends in order of preference:
# 1) sentence-transformers (fast & local)
# 2) langchain HuggingFaceEmbeddings (if user has langchain setup)
# 3) sklearn TF-IDF (fallback, text-based)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CSV_PATH = "Library_data.csv"

st.set_page_config(page_title="📚 Library Assistant (Local)", layout="centered")
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>📚 Library Assistant MMEC — Local</h1>", unsafe_allow_html=True)
st.write("This version uses local retrieval (no Chroma / Rust / external LLM). Works offline for small CSVs.")

# ---------------------------
# 0) Embedding utility wrappers
# ---------------------------
def try_sentence_transformers(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)
        def embed_texts(texts: List[str]) -> np.ndarray:
            arr = m.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr.astype(np.float32)
        return embed_texts
    except Exception:
        return None

def try_langchain_hf(model_name: str):
    try:
        # try multiple possible imports for compatibility
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            hf = HuggingFaceEmbeddings(model_name=model_name)
            def embed_texts(texts: List[str]) -> np.ndarray:
                lst = [hf.embed_query(t) for t in texts]
                return np.array(lst, dtype=np.float32)
            return embed_texts
        except Exception:
            from langchain_huggingface import HuggingFaceEmbeddings
            hf = HuggingFaceEmbeddings(model_name=model_name)
            def embed_texts(texts: List[str]) -> np.ndarray:
                lst = [hf.embed_query(t) for t in texts]
                return np.array(lst, dtype=np.float32)
            return embed_texts
    except Exception:
        return None

def try_tfidf_fallback():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = None
        def make_and_embed(texts: List[str]) -> np.ndarray:
            nonlocal vectorizer
            if vectorizer is None:
                vectorizer = TfidfVectorizer(max_features=4096)
                X = vectorizer.fit_transform(texts)
            else:
                X = vectorizer.transform(texts)
            return X.toarray().astype(np.float32)
        return make_and_embed
    except Exception:
        return None

# choose embedding function
embed_fn = None
embed_fn = try_sentence_transformers(EMBED_MODEL) or try_langchain_hf(EMBED_MODEL) or try_tfidf_fallback()
if embed_fn is None:
    st.error("No suitable embedding backend found. Please install one of: sentence-transformers OR langchain_huggingface/langchain_community OR scikit-learn.")
    st.stop()

# ---------------------------
# 1) Load CSV -> documents
# ---------------------------
@st.cache_data(show_spinner=False)
def load_csv_to_docs(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Place your CSV in the app folder.")
    df = pd.read_csv(path)
    # map common column names; adjust if your CSV has different names
    col_map = {
        "Acc. No.": "AccessionNo",
        "Acc No": "AccessionNo",
        "Accession No": "AccessionNo",
        "AccessionNo": "AccessionNo",
        "Title": "Title",
        "Author": "Author",
        "Place": "Place",
        "Publisher": "Publisher",
        "Year": "Year",
        "Edition": "Edition",
        "Language": "Language",
        "Subject": "Subject",
    }
    # normalize columns that exist
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}).fillna("")
    docs = []
    for _, row in df.iterrows():
        metadata = {
            "AccessionNo": str(row.get("AccessionNo", "")).strip(),
            "Title": str(row.get("Title", "")).strip(),
            "Author": str(row.get("Author", "")).strip(),
            "Edition": str(row.get("Edition", "")).strip(),
            "Language": str(row.get("Language", "")).strip(),
            "Subject": str(row.get("Subject", "")).strip(),
            "Year": str(row.get("Year", "")).strip(),
        }
        # Build a content string for embedding & retrieval
        content_parts = []
        for k in ["Title", "Author", "Edition", "Language", "Subject", "Year", "AccessionNo"]:
            v = metadata.get(k, "")
            if v:
                content_parts.append(f"{k}: {v}")
        content = " | ".join(content_parts)
        docs.append({"content": content, "metadata": metadata})
    return docs

docs = load_csv_to_docs(CSV_PATH)
if len(docs) == 0:
    st.warning("No documents found in CSV. Check the CSV path or its content.")
    st.stop()

# ---------------------------
# 2) Build embeddings for docs
# ---------------------------
@st.cache_resource(show_spinner=False)
def build_doc_embeddings(_docs):
    texts = [d["content"] for d in _docs]
    emb = embed_fn(texts)
    # normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    emb_norm = emb / norms
    return emb_norm

doc_embeddings = build_doc_embeddings(docs)

# ---------------------------
# 3) Similarity search helper
# ---------------------------
def cosine_search(query: str, k: int = 5):
    q_emb = embed_fn([query])
    # normalize
    q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_norm[q_norm==0] = 1.0
    q_emb = q_emb / q_norm
    # compute cosine similarities
    sims = (doc_embeddings @ q_emb.T).squeeze()  # shape (n_docs,)
    top_idx = np.argsort(-sims)[:k]
    return [{"score": float(sims[i]), "doc": docs[i]} for i in top_idx]

# ---------------------------
# 4) Formatter (exact required format)
# ---------------------------
def format_answer_from_metadata(metadata: dict):
    # If metadata fields are empty, use "Information not available"
    def field(key):
        v = metadata.get(key, "")
        return v if v and str(v).strip() else "Information not available"
    # exact format required by user
    out = [
        f"The Accession no of book is: {field('AccessionNo')}",
        f"The Title is: {field('Title')}",
        f"The Author is: {field('Author')}",
        f"The Edition is: {field('Edition')}",
        f"The Language is: {field('Language')}",
        f"The Subject is: {field('Subject')}",
        f"The Year of Publishing: {field('Year')}",
    ]
    return "\n".join(out)

# ---------------------------
# 5) Streamlit UI & query handling
# ---------------------------
st.write("Ask about books (title, author, subject, year, accession no). This assistant finds the best single match and outputs the exact required format.")

query = st.text_input("Enter your question:", placeholder="e.g. Which book is beginner friendly for physics? Or: accession no 12345")
k = st.slider("Number of candidates to consider (k)", min_value=1, max_value=10, value=5)

if query:
    with st.spinner("Searching library..."):
        results = cosine_search(query, k=k)
    # take best match
    best = results[0]
    score = best["score"]
    matched_doc = best["doc"]
    # If similarity score is low, treat as no relevant context
    # Threshold is adjustable; for TF-IDF it may need tuning
    threshold = 0.15  # safe low threshold; change if you want stricter matching
    if score < threshold:
        answer_text = "Information not available."
        st.subheader("Answer")
        st.text(answer_text)
    else:
        answer_text = format_answer_from_metadata(matched_doc["metadata"])
        st.subheader("Answer")
        st.text(answer_text)

    # show sources expandable
    with st.expander("📌 Sources (top candidates)"):
        for i, r in enumerate(results, 1):
            doc = r["doc"]
            sc = r["score"]
            md = doc["metadata"]
            st.markdown(f"**Source {i} — score: {sc:.4f}**")
            # show concise content and metadata
            st.text(doc["content"])
            st.write(md)
