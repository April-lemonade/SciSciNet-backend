# SciSciNet Backend

## Requirements
- Python 3.9+
- fastapi
- uvicorn[standard]
- duckdb
- pandas
- pyarrow
- requests

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
Before starting the backend, generate the parquet files by running the notebook:

1) Create an environment file with your Hugging Face token (used by the notebook):
```bash
cp env.example .env
# edit .env and set HF_TOKEN=...
```

2) Open and run `data_processing.ipynb` top-to-bottom. It will download/process raw data and export the required parquet files under `dataset/` (e.g., `dataset/sample/`).

3) Verify the files exist (examples):
```
dataset/sample/citations_5y_internal.parquet
dataset/sample/papers_university_5y_cs.parquet
dataset/sample/coauthor_details_5y.parquet
dataset/sample/authors_university_5y_cs.parquet
dataset/univ_cs_10y_papers.parquet
```

## Project Structure
```
SciSciNet-backend/
  ├─ dataset/                 # Parquet data files (papers, citations, authors, etc.)
  │   └─ sample/              # Sample subset used by the app by default
  ├─ main.py                  # FastAPI app and API routes
  └─ data_processing.ipynb    # Data processing / export notebook
```

## Running the Server
From this directory:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --env-file env.example
```
The frontend expects the backend at `http://localhost:8000` by default.

Ensure the parquet files referenced by `main.py` exist under `dataset/` with the expected filenames.
