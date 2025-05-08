import json
import time
import pandas as pd
from utils.semantic_search import search as faiss_search
from utils.lexical_search import search_bm25, search_tfidf
from app import search_text_pg


with open("data/gold_test_set.json", "r") as f:
    gold = json.load(f)


dispatch = {
    "FAISS": lambda q: faiss_search(q, top_k=1),
    "TF-IDF": lambda q: [s for s, _ in search_tfidf(q, top_k=1)],
    "BM25": lambda q: [s for s, _ in search_bm25(q, top_k=1)],
    "IVFFLAT": lambda q: search_text_pg(q, table_name="video_segments_ivfflat", top_k=1),
    "HNSW": lambda q: search_text_pg(q, table_name="video_segments_hnsw", top_k=1)
}

summary_rows = []
detailed_logs = []

for method, func in dispatch.items():
    correct = 0
    rejected = 0
    total_answerable = 0
    latencies = []

    for entry in gold:
        q = entry["question"]
        expected_start = entry.get("ground_truth_start")
        expected_end = entry.get("ground_truth_end")
        answerable = entry["answerable"]

        start = time.time()
        retrieved = func(q)
        end = time.time()
        latency = end - start
        latencies.append(latency)

        record = {
            "Method": method,
            "Question": q,
            "Answerable": answerable,
            "Latency (s)": round(latency, 3),
            "Retrieved": bool(retrieved),
            "Status": ""
        }

        if not retrieved:
            if not answerable:
                rejected += 1
                record["Status"] = "Correctly Rejected"
            else:
                record["Status"] = "Missed Answer"
        else:
            if answerable:
                total_answerable += 1
                top = retrieved[0]
                pred_start = top["start"]
                if expected_start - 5 <= pred_start <= expected_end + 5:
                    correct += 1
                    record["Status"] = "Correct"
                else:
                    record["Status"] = "Wrong Timestamp"
            else:
                record["Status"] = "False Positive"

        detailed_logs.append(record)

    accuracy_str = f"{correct}/{total_answerable}" if total_answerable else "N/A"
    rejection_str = f"{rejected}/5"
    avg_latency = f"{sum(latencies) / len(latencies):.3f}"

    summary_rows.append({
        "Method": method,
        "Accuracy (Answerable)": accuracy_str,
        "Rejection (Unanswerable)": rejection_str,
        "Avg Latency (s)": avg_latency
    })


summary_df = pd.DataFrame(summary_rows)
print("\n=== Evaluation Summary Table ===\n")
print(summary_df.to_markdown(index=False))


summary_df.to_markdown("evaluation_summary.md", index=False)
print("Saved summary to evaluation_summary.md")


pd.DataFrame(detailed_logs).to_csv("evaluation_detailed_log.csv", index=False)
print("Saved detailed log to evaluation_detailed_log.csv")
