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

## How the Data Is Processed
The notebook `data_processing.ipynb` prepares compact parquet datasets used by the API:

1) Source acquisition
   - Authenticates with Hugging Face via `HF_TOKEN`.
   - Downloads raw parquet files (papers, citations, authors/coauthors).

2) Paper subset (university CS, last 5–10 years)
   - Filters papers to Computer Science by field tags.
   - Restricts to university-affiliated works (based on affiliation data).
   - Limits to a recent-year window for faster visualization.
   - Exports: `papers_university_5y_cs.parquet` (core paper attributes).

3) Internal citation edges
   - Builds citation edges where both source/target are in the university CS paper subset.
   - Keeps a year-filtered window to manage graph size.
   - Exports: `citations_5y_internal.parquet` (columns: `citing_paperid`, `cited_paperid`, `year`).

4) Coauthor relationships
   - Expands author–paper rows; pairs authors who coauthored the same paper.
   - Deduplicates pairs and aggregates by (source, target) with counts and year.
   - Exports: `coauthor_details_5y.parquet` (normalized rows to derive pairs) and `authors_university_5y_cs.parquet` (author metadata).

5) Dashboard aggregates (10-year window)
   - Aggregates per-year paper counts and patent counts for the timeline and histogram.
   - Exports: `univ_cs_10y_papers.parquet` (columns include `year`, `Patent_Count`).

These outputs are what `main.py` reads to serve the networks and dashboard endpoints.

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
