import streamlit as st
import psycopg2
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import clip
from PIL import Image

from utils.semantic_search import search as faiss_search
from utils.lexical_search import search_tfidf, search_bm25

st.set_page_config(page_title="Multimodal RAG Chat")

DB_CONFIG = {
    'dbname': 'rag_db',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}

@st.cache_resource
def connect_db():
    return psycopg2.connect(**DB_CONFIG)

conn = connect_db()
cursor = conn.cursor()

@st.cache_resource
def load_text_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_image_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

text_model = load_text_model()
clip_model, clip_preprocess, clip_device = load_image_model()

def format_time(seconds):
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"



def search_text_pg(query, table_name="video_segments", top_k=5, threshold=0.92):
    query_emb = text_model.encode([query])[0].tolist()
    cursor.execute(f"""
        SELECT id, start_time, end_time, text, image_path, text_embedding <-> %s::vector AS distance
        FROM {table_name}
        ORDER BY distance ASC
        LIMIT %s
    """, (query_emb, top_k))
    rows = cursor.fetchall()

    if not rows or rows[0][-1] > threshold:
        return []

    return [
        {"start": r[1], "end": r[2], "text": r[3], "image": r[4]}
        for r in rows
    ]


def search_image_pg(image, table_name="video_segments", top_k=5, threshold=22):
    image_input = clip_preprocess(image).unsqueeze(0).to(clip_device)
    with torch.no_grad():
        image_emb = clip_model.encode_image(image_input)[0].cpu().numpy().tolist()
    cursor.execute(f"""
        SELECT id, start_time, end_time, text, image_path,
               image_embedding <-> %s::vector AS score
        FROM {table_name}
        ORDER BY score
        LIMIT %s
    """, (image_emb, top_k))
    results = cursor.fetchall()

    if not results or results[0][-1] > threshold:
        return []

    return [{"start": r[1], "end": r[2], "text": r[3], "image": r[4]} for r in results]



st.title("Video Q&A Chat Interface")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

retrieval_method = st.selectbox("Retrieval Method", [
    "FAISS Flat (semantic dense retrieval)",
    "pgvector IVFFLAT (semantic clustered retrieval)",
    "pgvector HNSW (semantic graph retrieval)",
    "TF-IDF (lexical keyword retrieval)",
    "BM25 (lexical probabilistic retrieval)"
])

with st.container():
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input("Ask a question", placeholder="Enter your question here", label_visibility="collapsed")
    with col2:
        submit_text = st.button("→", use_container_width=True)
        st.write("")


if submit_text and user_input:
    if retrieval_method == "FAISS Flat (semantic dense retrieval)":
        results = faiss_search(user_input, top_k=5)
    elif retrieval_method == "pgvector IVFFLAT (semantic clustered retrieval)":
        results = search_text_pg(user_input, table_name="video_segments_ivfflat")
    elif retrieval_method == "pgvector HNSW (semantic graph retrieval)":
        results = search_text_pg(user_input, table_name="video_segments_hnsw")
    elif retrieval_method == "TF-IDF (lexical keyword retrieval)":
        results = [{"start": s.get("start", 0), "end": s.get("end", 0), "text": s["text"], "image": s.get("image_path")} for s, _ in search_tfidf(user_input)]
    elif retrieval_method == "BM25 (lexical probabilistic retrieval)":
        results = [{"start": s.get("start", 0), "end": s.get("end", 0), "text": s["text"], "image": s.get("image_path")} for s, _ in search_bm25(user_input)]

    st.session_state.chat_history.append({
        "query": user_input,
        "results": results,
        "method": retrieval_method,
        "type": "text"
    })


if retrieval_method in ["pgvector IVFFLAT (semantic clustered retrieval)", "pgvector HNSW (semantic graph retrieval)"]:
    uploaded_file = st.file_uploader("Upload an image for visual search", type=["jpg", "png"])
    if uploaded_file:
        if st.button("Search with Image"):
            image = Image.open(uploaded_file)
            table = "video_segments_ivfflat" if "IVFFLAT" in retrieval_method else "video_segments_hnsw"
            results = search_image_pg(image, table_name=table)
            st.session_state.chat_history.append({
                "query": uploaded_file.name,
                "results": results,
                "method": retrieval_method,
                "type": "image",
                "image_preview": image
            })


for turn in reversed(st.session_state.chat_history):
    if turn["type"] == "text":
        st.markdown(f"**You:** {turn['query']}")
    else:
        st.markdown(f"**You (image):** {turn['query']}")
        st.image(turn.get("image_preview"))

    st.markdown(f"*Retrieved with:* `{turn['method']}`")
    if turn["results"]:
        for r in turn["results"]:
            st.video("data/video.mp4", start_time=int(r['start']))
            st.markdown(f"`{format_time(r['start'])}` → `{format_time(r['end'])}`")
            st.markdown(f"{r['text']}")
            st.markdown("---")
    else:
        st.warning("No relevant segment found.Try rephrasing your question.")
