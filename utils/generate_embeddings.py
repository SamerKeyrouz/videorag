import json
import os
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import clip

SEGMENTS_PATH = "../data/segments.json"
TEXT_EMB_PATH = "../data/text_embeddings.npy"
IMG_EMB_PATH = "../data/image_embeddings.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_segments():
    with open(SEGMENTS_PATH, 'r') as f:
        return json.load(f)


def embed_text(segments):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    texts = [seg["text"] for seg in segments]
    print(f"Embedding {len(texts)} text segments...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    np.save(TEXT_EMB_PATH, embeddings)
    print(f"Text embeddings saved to {TEXT_EMB_PATH}")
    return embeddings


def embed_images(segments):
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    embeddings = []
    print(f"Embedding {len(segments)} keyframes...")
    for seg in segments:
        img_path = os.path.abspath(os.path.join(os.path.dirname(SEGMENTS_PATH), seg["image_path"]))
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(image)
        emb = emb.cpu().numpy().squeeze()
        embeddings.append(emb)
    embeddings = np.stack(embeddings)
    np.save(IMG_EMB_PATH, embeddings)
    print(f"Image embeddings saved to {IMG_EMB_PATH}")
    return embeddings


if __name__ == "__main__":
    segments = load_segments()
    embed_text(segments)
    embed_images(segments)
    print("All embeddings complete!")
