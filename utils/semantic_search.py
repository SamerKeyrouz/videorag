import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer

SEGMENT_PATH = "data/segments.json"
EMBEDDINGS_PATH = "data/text_embeddings.npy"


with open(SEGMENT_PATH, "r", encoding="utf-8") as f:
    segments = json.load(f)
text_embeddings = np.load(EMBEDDINGS_PATH).astype("float32")


dimension = text_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(text_embeddings)


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def search(query, top_k=3, threshold=0.6):
    query_vec = model.encode([query]).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score > threshold:
            continue
        seg = segments[idx]
        results.append({
            "score": float(score),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "image": seg["image_path"]
        })
    return results



if __name__ == "__main__":
    query = "What is token jumping?"
    results = search(query, top_k=5)
    print("Results for:", query)
    for res in results:
        print(f"\n {res['start']}s → {res['end']}s")
        print(f"{res['text']}")
        print(f"Keyframe: {res['image']}")
