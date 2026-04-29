import streamlit as st
import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.title("Local RAG (Optimized for 8GB RAM)")

# モデルのロードをキャッシュ
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# PDF処理とインデックス構築をセッションで保持（再計算を防ぐ）
def process_pdf(file):
    if 'index' not in st.session_state or st.session_state.get('last_file') != file.name:
        with st.spinner("Processing PDF and building index..."):
            reader = PdfReader(file)
            text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
            
            # チャンク分割
            size, overlap = 300, 80
            chunks = [text[i:i+size] for i in range(0, len(text), size - overlap)]
            
            # ベクトル化
            vecs = embedder.encode(chunks, convert_to_numpy=True)
            index = faiss.IndexFlatL2(vecs.shape[1])
            index.add(vecs.astype("float32"))
            
            # セッションに保存
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.last_file = file.name

def ask_llm(q, ctx):
    # システムプロンプトを追加して、要約癖を直し、回答の質を上げる
    system_prompt = "You are a helpful assistant. Use only the provided context to answer the question briefly. If the answer is not in the context, say I don't know."
    full_prompt = f"{system_prompt}\n\nContext:\n{ctx}\n\nQuestion: {q}"

    try:
        r = requests.post(
            "http://ollama:11434/api/generate",
            json={
                "model": "phi", # または "mistral"
                "prompt": full_prompt,
                "stream": False
            },
            timeout=120 # 8GB Macのスワップ待ち用に長めに設定
        )
        return r.json().get("response", "No response error")
    except Exception as e:
        return f"Connection Error: {e}"

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    process_pdf(file) # ここで初回のみインデックス作成
    q = st.text_input("Ask a question about the PDF")

    if q and 'index' in st.session_state:
        # 検索
        q_vec = embedder.encode([q], convert_to_numpy=True).astype("float32")
        _, I = st.session_state.index.search(q_vec, 3)
        ctx = [st.session_state.chunks[i] for i in I[0]]
        
        with st.spinner("Thinking..."):
            ans = ask_llm(q, "\n".join(ctx))

        st.subheader("Answer")
        st.write(ans)

        with st.expander("Reference Chunks"):
            for c in ctx:
                st.write(c)