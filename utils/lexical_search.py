import json
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import numpy as np
import nltk

nltk.download("punkt")

SEGMENTS_PATH = "data/segments.json"

with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
    segments = json.load(f)

texts = [seg["text"] for seg in segments]


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)


tokenized_corpus = [word_tokenize(doc.lower()) for doc in texts]
bm25 = BM25Okapi(tokenized_corpus)


def search_tfidf(query, top_k=5, threshold=0.35):
    query_vec = tfidf_vectorizer.transform([query])
    scores = np.dot(tfidf_matrix, query_vec.T).toarray().squeeze()

    if len(scores) == 0 or np.max(scores) < threshold:
        return []

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(segments[i], scores[i]) for i in top_indices]

def search_bm25(query, top_k=5, threshold=12):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1]

    if scores[top_indices[0]] < threshold:
        return []

    return [(segments[i], scores[i]) for i in top_indices[:top_k]]



if __name__ == "__main__":
    q = "What is token jumping?"

    print("TF-IDF Results:")
    for seg, score in search_tfidf(q):
        print(f"\n{seg['start']}s → {seg['end']}s")
        print(f"{seg['text']}")
        print(f"Score: {score:.4f}")

    print("\nBM25 Results:")
    for seg, score in search_bm25(q):
        print(f"\n{seg['start']}s → {seg['end']}s")
        print(f"{seg['text']}")
        print(f"Score: {score:.4f}")
