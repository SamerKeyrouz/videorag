# Multimodal Video Question Answering (RAG System)

This project implements a multimodal **Retrieval-Augmented Generation (RAG)** system that allows users to ask natural language questions about a YouTube video. The system retrieves relevant video segments using both **semantic** (dense) and **lexical** (sparse) methods, and supports **textual** and **visual** queries.

The demo is built around the talk:
[Parameterized Complexity of Token Sliding, Token Jumping – Amer Mouawad](https://www.youtube.com/watch?v=dARr3lGKwk8)

---

## Features

- Whisper-based **speech-to-text** transcription
- Transcript segmentation with aligned keyframes
- Text embeddings using Sentence Transformers
- Image embeddings using OpenAI CLIP
- **Semantic retrieval** via:
  - FAISS
  - pgvector IVFFLAT
  - pgvector HNSW
- **Lexical retrieval** via:
  - TF-IDF
  - BM25
- Interactive **Streamlit** app with video playback and fallback response

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone 

```
### 2. Obtain the video (Required for Full Pipeline)
The demonstration video is excluded from the repository due to size constraints. To recreate the full pipeline, download the source video from youtube (https://www.youtube.com/watch?v=dARr3lGKwk8) and place it under data directory. 

### 3. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

---

### 4. PostgreSQL Setup
Install PostgreSQL and pgvector extension. Then create the database `rag_db` and run the following schema in pgAdmin or psql:

### Table + Indexes
```sql
-- Table: public.video_segments_hnsw
CREATE TABLE IF NOT EXISTS public.video_segments_hnsw (
    id integer NOT NULL DEFAULT nextval('video_segments_hnsw_id_seq'::regclass),
    start_time double precision,
    end_time double precision,
    text text COLLATE pg_catalog."default",
    image_path text COLLATE pg_catalog."default",
    text_embedding vector(384),
    image_embedding vector(512),
    CONSTRAINT video_segments_hnsw_pkey PRIMARY KEY (id)
);

ALTER TABLE IF EXISTS public.video_segments_hnsw
    OWNER to postgres;

-- Indexes
CREATE INDEX IF NOT EXISTS hnsw_image_idx
    ON public.video_segments_hnsw USING hnsw
    (image_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS hnsw_text_idx
    ON public.video_segments_hnsw USING hnsw
    (text_embedding vector_cosine_ops);
```


Before running `load_to_pg.py`, make sure to update database credentials in the file

---

## Reproducibility
### Option 1: Re-run Entire Pipeline (Requires local video under data directory)
Use this to test with a **new video**:
```bash
python transcribe_audio.py
python process_segments.py
python generate_embeddings.py
python load_to_pg.py
```

### Option 2: Use Precomputed Files
If the `data/` folder is included with transcript, segments, and embeddings:
```bash
python load_to_pg.py
streamlit run app.py
```

---

## Launch the Interface
```bash
streamlit run app.py
```

You can now:
- Select a retrieval method (FAISS, pgvector, TF-IDF, BM25)
- Ask a question or upload an image (for visual search)
- View the retrieved video clip with text and timestamps

---

## Evaluation
To run the evaluation on the gold test set:
```bash
python evaluate_retrieval.py
```
Generates:
- Accuracy, rejection rate, and latency for each method
- Saves detailed results to `evaluation_detailed_log.csv`

---

## Notes
- Built and tested with Python 3.10+
- Ensure `ffmpeg` is installed for MoviePy and Whisper to function
- PostgreSQL with pgvector must be set up if using pgvector-based retrieval

---

## Contact

For questions, suggestions, or improvements, feel free to reach out via:

- Email: [ssk45@mail.aub.edu](mailto:ssk45@mail.aub.edu)  
- Mobile: +961 76 529 820
---

**Author:** Samer Keyrouz

---

**Video used:** [Token Jumping & Sliding Talk](https://www.youtube.com/watch?v=dARr3lGKwk8)