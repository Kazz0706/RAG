# app.py
# Minimal RAG in one file (2-hour version)
# Run:
#   pip install streamlit sentence-transformers faiss-cpu pypdf transformers torch
#   streamlit run app.py

import streamlit as st
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ----------------------------
# Load Models (cached)
# ----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

embedder = load_embedder()
generator = load_generator()

# ----------------------------
# Helpers
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def chunk_text(text, chunk_size=300, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_faiss_index(chunks):
    vectors = embedder.encode(chunks, convert_to_numpy=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors.astype("float32"))
    return index, vectors

def retrieve(query, chunks, index, top_k=3):
    qvec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(qvec, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

def answer_question(query, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""
Answer the question using only the context below.

Context:
{context_text}

Question:
{query}

Answer:
"""
    result = generator(prompt)[0]["generated_text"]
    return result

# ----------------------------
# UI
# ----------------------------
st.title("📄 Minimal RAG PDF Chatbot")
st.write("Upload a PDF, ask questions, and get answers.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    if len(text.strip()) == 0:
        st.error("Could not extract text from PDF.")
        st.stop()

    chunks = chunk_text(text)
    index, _ = build_faiss_index(chunks)

    st.success(f"Loaded PDF with {len(chunks)} chunks.")

    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Searching and generating answer..."):
            contexts = retrieve(query, chunks, index, top_k=3)
            answer = answer_question(query, contexts)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(c)