import psycopg2
import json
import numpy as np


# Please fill in your PostgreSQL credentials below before running

conn = psycopg2.connect(
    dbname="enter your db name here",
    user="enter your user here",
    password="enter your password here",
    host="enter hostname",
    port="enter portnb"
)
cursor = conn.cursor()


print("Loading segments...")
with open("data/segments.json", "r", encoding="utf-8") as f:
    segments = json.load(f)

text_embeddings = np.load("data/text_embeddings.npy")
image_embeddings = np.load("data/image_embeddings.npy")

print("Inserting segments into video_segments_ivfflat...")
for i, segment in enumerate(segments):
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    image_path = segment["image_path"]
    text_emb = text_embeddings[i].tolist()
    image_emb = image_embeddings[i].tolist()

    cursor.execute("""
        INSERT INTO video_segments_ivfflat (
            start_time, end_time, text, image_path, text_embedding, image_embedding
        ) VALUES (%s, %s, %s, %s, %s, %s)
    """, (start, end, text, image_path, text_emb, image_emb))

print("Inserting segments into video_segments_hnsw...")
for i, segment in enumerate(segments):
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    image_path = segment["image_path"]
    text_emb = text_embeddings[i].tolist()
    image_emb = image_embeddings[i].tolist()

    cursor.execute("""
        INSERT INTO video_segments_hnsw (
            start_time, end_time, text, image_path, text_embedding, image_embedding
        ) VALUES (%s, %s, %s, %s, %s, %s)
    """, (start, end, text, image_path, text_emb, image_emb))


conn.commit()
cursor.close()
conn.close()
print("All segments inserted into both tables!")
