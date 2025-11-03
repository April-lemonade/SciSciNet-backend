# main.py
from typing import Optional, List, Dict, Any
import math
import numpy as np
import pandas as pd
import os
import duckdb
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import requests

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "dataset/sample"
PARQUET_CITATIONS = os.path.join(DATA_DIR, "citations_5y_internal.parquet")
PARQUET_PAPERS = os.path.join(DATA_DIR, "papers_university_5y_cs.parquet")
PARQUET_COAUTHORS = os.path.join(DATA_DIR, "coauthor_details_5y.parquet")
PARQUET_AUTHORS = os.path.join(DATA_DIR, "authors_university_5y_cs.parquet")
PARQUET_DASHBOARD = os.path.join(DATA_DIR, "univ_cs_10y_papers.parquet")

REQUIRED_FILES = [PARQUET_CITATIONS, PARQUET_PAPERS, PARQUET_COAUTHORS, PARQUET_AUTHORS]

app = FastAPI(title="Citation Network Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# DuckDB connection (thread-safe)
# -----------------------------
def get_con() -> duckdb.DuckDBPyConnection:
    for f in REQUIRED_FILES:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    # Create a new connection for each request to avoid mutex issues
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE OR REPLACE VIEW citations AS
        SELECT citing_paperid AS source, cited_paperid AS target, year
        FROM read_parquet('{PARQUET_CITATIONS}');
    """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW papers AS
        SELECT *
        FROM read_parquet('{PARQUET_PAPERS}');
    """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW coauthor_details AS
        SELECT paperid, authorid, author_position, year
        FROM read_parquet('{PARQUET_COAUTHORS}');
    """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW authors AS
        SELECT *
        FROM read_parquet('{PARQUET_AUTHORS}');
    """
    )
    return con


def to_records(df) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


# -----------------------------
# OpenAlex enrichment (optional)
# -----------------------------
OPENALEX_ENABLE = os.environ.get("OPENALEX_ENABLE", "true").lower() in {
    "1",
    "true",
    "yes",
}
OPENALEX_TIMEOUT = float(os.environ.get("OPENALEX_TIMEOUT", "6"))


def _build_abstract_from_inverted_index(
    inv_idx: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Rebuild abstract text from OpenAlex inverted index format."""
    if not inv_idx or not isinstance(inv_idx, dict):
        return None
    try:
        positions: Dict[int, str] = {}
        for token, idxs in inv_idx.items():
            for i in idxs:
                positions[int(i)] = token
        if not positions:
            return None
        tokens = [positions[i] for i in sorted(positions.keys())]
        return " ".join(tokens)
    except Exception:
        return None


def fetch_openalex_details(
    paper_id: Optional[str], doi: Optional[str]
) -> Dict[str, Any]:
    if not OPENALEX_ENABLE:
        return {}
    base = "https://api.openalex.org/works/"
    url: Optional[str] = None
    if paper_id and isinstance(paper_id, str) and paper_id.startswith("W"):
        url = base + paper_id
    elif doi and isinstance(doi, str) and len(doi) > 0:
        # ensure DOI format as OpenAlex expects
        doi_suffix = doi.replace("https://doi.org/", "")
        url = base + f"https://doi.org/{doi_suffix}"
    if not url:
        return {}
    try:
        r = requests.get(url, timeout=OPENALEX_TIMEOUT)
        if r.status_code != 200:
            return {}
        data = r.json()
        result: Dict[str, Any] = {}
        title = data.get("title")
        if isinstance(title, str) and title:
            result["title"] = title
        host = data.get("host_venue") or {}
        if isinstance(host, dict):
            vname = host.get("display_name")
            if isinstance(vname, str) and vname:
                result["venue"] = vname
        authorships = data.get("authorships") or []
        if isinstance(authorships, list) and authorships:
            names: List[str] = []
            for a in authorships:
                if isinstance(a, dict):
                    author = a.get("author")
                    if isinstance(author, dict):
                        nm = author.get("display_name")
                        if isinstance(nm, str) and nm:
                            names.append(nm)
            if names:
                result["authors"] = ", ".join(names)
        abs_text = _build_abstract_from_inverted_index(
            data.get("abstract_inverted_index")
        )
        if abs_text:
            result["abstract"] = abs_text
        return result
    except Exception:
        return {}


# -----------------------------
# Routes
# -----------------------------
@app.get("/api/network/citations")
def citation_network(
    year_from: int = Query(2020, ge=0),
    year_to: Optional[int] = Query(None, ge=0),
    max_edges: int = Query(5000, ge=1, le=200000),
    min_degree: int = Query(
        0, ge=0, description="Filter out nodes with degree less than this value"
    ),
    include_isolated: bool = Query(
        False, description="Whether to include isolated nodes"
    ),
):
    """
    Returns D3-compatible node/link structure:
    {
      "nodes": [{id, year, doctype, team_size, indegree, outdegree, degree}, ...],
      "links": [{source, target, year}, ...]
    }
    Contains only internal citations within the university CS papers (both ends are university CS papers).
    """
    con = get_con()

    where = "WHERE year >= ?"
    params: List[Any] = [year_from]
    if year_to is not None:
        where += " AND year <= ?"
        params.append(year_to)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _links AS
        SELECT source, target, year
        FROM citations
        {where}
        ORDER BY year DESC, source, target
        LIMIT {max_edges}
    """,
        params,
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _deg AS
        WITH outd AS (
            SELECT source AS id, COUNT(*) AS outdegree FROM _links GROUP BY 1
        ),
        ind AS (
            SELECT target AS id, COUNT(*) AS indegree FROM _links GROUP BY 1
        )
        SELECT
            COALESCE(outd.id, ind.id) AS id,
            COALESCE(ind.indegree, 0)  AS indegree,
            COALESCE(outd.outdegree, 0) AS outdegree,
            COALESCE(ind.indegree, 0) + COALESCE(outd.outdegree, 0) AS degree,
            COALESCE(ind.indegree, 0) AS network_cited_by_count
        FROM outd
        FULL OUTER JOIN ind ON outd.id = ind.id
    """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _used AS
        SELECT source AS paperid FROM _links
        UNION
        SELECT target AS paperid FROM _links
    """
    )

    if include_isolated:
        iso_where = "WHERE year >= ?"
        iso_params: List[Any] = [year_from]
        if year_to is not None:
            iso_where += " AND year <= ?"
            iso_params.append(year_to)

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE _iso_nodes AS
            SELECT paperid AS id, year, doctype, team_size, cited_by_count
            FROM papers
            {iso_where}
            EXCEPT
            SELECT p.paperid AS id, p.year, p.doctype, p.team_size, p.cited_by_count
            FROM papers p JOIN _used u ON p.paperid = u.paperid
        """,
            iso_params,
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE _nodes AS
            SELECT
                u.paperid AS id, p.year, p.doctype, p.team_size, p.cited_by_count,
                d.indegree, d.outdegree, d.degree, d.network_cited_by_count
            FROM _used u
            JOIN papers p ON u.paperid = p.paperid
            LEFT JOIN _deg d ON u.paperid = d.id

            UNION ALL
            SELECT id, year, doctype, team_size, cited_by_count, 0, 0, 0, 0
            FROM _iso_nodes
        """
        )
    else:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE _nodes AS
            SELECT
                u.paperid AS id,
                p.year,
                p.date,
                p.doctype,
                p.team_size,
                p.cited_by_count,
                COALESCE(d.indegree,0) AS indegree,
                COALESCE(d.outdegree,0) AS outdegree,
                COALESCE(d.degree,0) AS degree,
                COALESCE(d.network_cited_by_count,0) AS network_cited_by_count
            FROM _used u
            JOIN papers p ON u.paperid = p.paperid
            LEFT JOIN _deg d ON u.paperid = d.id
        """
        )

    if min_degree > 0:
        con.execute(
            "CREATE OR REPLACE TEMP TABLE _nodes AS SELECT * FROM _nodes WHERE degree >= ?",
            [min_degree],
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE _links AS
            SELECT l.*
            FROM _links l
            JOIN _nodes n1 ON l.source = n1.id
            JOIN _nodes n2 ON l.target = n2.id
        """
        )

    nodes_df = con.execute(
        """
        SELECT id, year, date, doctype, team_size, cited_by_count, network_cited_by_count, indegree, outdegree, degree
        FROM _nodes
        ORDER BY degree DESC, year DESC
    """
    ).fetchdf()
    links_df = con.execute(
        """
        SELECT source, target, year
        FROM _links
        ORDER BY year DESC
    """
    ).fetchdf()

    return {
        "nodes": to_records(nodes_df),
        "links": to_records(links_df),
        "meta": {
            "year_from": year_from,
            "year_to": year_to,
            "max_edges": max_edges,
            "min_degree": min_degree,
            "include_isolated": include_isolated,
            "node_count": int(nodes_df.shape[0]),
            "link_count": int(links_df.shape[0]),
        },
    }


@app.get("/api/network/coauthors")
def coauthor_network(
    year_from: int = Query(2020, ge=0),
    year_to: Optional[int] = Query(None, ge=0),
    max_edges: int = Query(5000, ge=1, le=200000),
    min_degree: int = Query(
        0, ge=0, description="Filter out authors with degree less than this value"
    ),
    include_isolated: bool = Query(
        False, description="Whether to include isolated authors"
    ),
):
    """
    Returns author coauthor network in D3-compatible structure:
    {
      "nodes": [{id, author_name, institution_name, works_count, cited_by_count, degree, coauthored_count}, ...],
      "links": [{source, target, weight, year}, ...]
    }
    Edges are undirected, deduplicated using source<target; weight is number of coauthored papers.
    """
    con = get_con()

    where = "WHERE year >= ?"
    params: List[Any] = [year_from]
    if year_to is not None:
        where += " AND year <= ?"
        params.append(year_to)

    # Select coauthor pairs (deduplicate: smaller authorid as source, larger as target)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _co AS
        SELECT paperid, authorid, year
        FROM coauthor_details
        {where}
    """,
        params,
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _pairs AS
        SELECT a1.authorid AS source, a2.authorid AS target, GREATEST(a1.year, a2.year) AS year
        FROM _co a1
        JOIN _co a2
          ON a1.paperid = a2.paperid
         AND a1.authorid < a2.authorid
    """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _links AS
        SELECT source, target, COUNT(*) AS weight, MAX(year) AS year
        FROM _pairs
        GROUP BY 1,2
        ORDER BY weight DESC
        LIMIT {max_edges}
    """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _deg AS
        WITH all_ids AS (
            SELECT source AS id FROM _links
            UNION
            SELECT target AS id FROM _links
        )
        SELECT id,
               COALESCE((
                   SELECT COUNT(*) FROM _links l WHERE l.source = id OR l.target = id
               ), 0) AS degree
        FROM all_ids
    """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _used AS
        SELECT source AS authorid FROM _links
        UNION
        SELECT target AS authorid FROM _links
    """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _nodes AS
        SELECT u.authorid AS id,
               a.author_name,
               a.institution_name,
               a.works_count,
               a.cited_by_count,
               d.degree,
               d.degree AS coauthored_count
        FROM _used u
        LEFT JOIN authors a ON u.authorid = a.authorid
        LEFT JOIN _deg d ON u.authorid = d.id
    """
    )

    if min_degree > 0:
        con.execute(
            "CREATE OR REPLACE TEMP TABLE _nodes AS SELECT * FROM _nodes WHERE degree >= ?",
            [min_degree],
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE _links AS
            SELECT l.*
            FROM _links l
            JOIN _nodes n1 ON l.source = n1.id
            JOIN _nodes n2 ON l.target = n2.id
        """
        )

    nodes_df = con.execute(
        """
        SELECT id, author_name, institution_name, works_count, cited_by_count, degree, coauthored_count
        FROM _nodes
        ORDER BY degree DESC, cited_by_count DESC NULLS LAST
    """
    ).fetchdf()
    links_df = con.execute(
        """
        SELECT source, target, year, weight
        FROM _links
        ORDER BY weight DESC
    """
    ).fetchdf()

    return {
        "nodes": to_records(nodes_df),
        "links": to_records(links_df),
        "meta": {
            "year_from": year_from,
            "year_to": year_to,
            "max_edges": max_edges,
            "min_degree": min_degree,
            "include_isolated": include_isolated,
            "node_count": int(nodes_df.shape[0]),
            "link_count": int(links_df.shape[0]),
        },
    }


@app.get("/api/papers/{paper_id}")
def paper_details(paper_id: str):
    """
    Return all details for a single paper from papers table plus degree metrics.
    """
    con = get_con()

    paper_df = con.execute(
        "SELECT * FROM papers WHERE paperid = ?", [paper_id]
    ).fetchdf()

    if paper_df.empty:
        return {"ok": False, "error": f"paper not found: {paper_id}"}

    raw_data = paper_df.iloc[0].to_dict()

    def sanitize_value(v):
        """Convert pandas/numpy types to Python native types, handling NaN/NA."""
        try:
            if v is None:
                return None
            if isinstance(v, float) and math.isnan(v):
                return None
            try:
                import pandas as pd  # type: ignore

                if isinstance(v, pd._libs.missing.NAType):
                    return None
            except Exception:
                pass
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                fv = float(v)
                return None if math.isnan(fv) else fv
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (np.ndarray,)):
                return v.tolist()
            return v
        except Exception:
            return str(v)

    paper_data: Dict[str, Any] = {k: sanitize_value(v) for k, v in raw_data.items()}

    indeg = con.execute(
        "SELECT COUNT(*) FROM citations WHERE target = ?", [paper_id]
    ).fetchone()[0]
    outdeg = con.execute(
        "SELECT COUNT(*) FROM citations WHERE source = ?", [paper_id]
    ).fetchone()[0]

    paper_data.update(
        {
            "ok": True,
            "indegree": int(indeg),
            "outdegree": int(outdeg),
            "degree": int(indeg + outdeg),
        }
    )

    # Enrich with OpenAlex data (title, venue, authors, abstract)
    try:
        raw_doi = paper_data.get("doi")
        doi_val = None
        if isinstance(raw_doi, str) and raw_doi:
            doi_val = raw_doi
        enrich = fetch_openalex_details(paper_id, doi_val)
        for k, v in enrich.items():
            if v is not None and not paper_data.get(k):
                paper_data[k] = v
    except Exception:
        pass
    return paper_data


@app.get("/api/authors/{author_id}")
def author_details(author_id: str):
    """
    Return all details for a single author from authors table.
    """
    con = get_con()

    author_df = con.execute(
        "SELECT * FROM authors WHERE authorid = ?", [author_id]
    ).fetchdf()

    if author_df.empty:
        return {"ok": False, "error": f"author not found: {author_id}"}

    raw_data = author_df.iloc[0].to_dict()

    def sanitize_value(v):
        """Convert pandas/numpy types to Python native types, handling NaN/NA."""
        try:
            if v is None:
                return None
            if isinstance(v, float) and math.isnan(v):
                return None
            try:
                import pandas as pd  # type: ignore

                if isinstance(v, pd._libs.missing.NAType):
                    return None
            except Exception:
                pass
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                fv = float(v)
                return None if math.isnan(fv) else fv
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (np.ndarray,)):
                return v.tolist()
            return v
        except Exception:
            return str(v)

    author_data: Dict[str, Any] = {k: sanitize_value(v) for k, v in raw_data.items()}

    author_data.update(
        {
            "ok": True,
        }
    )

    return author_data


@app.get("/api/dashboard/timeline")
def get_timeline_data(
    year_from: int = Query(2015, ge=2015, le=2025),
    year_to: int = Query(2025, ge=2015, le=2025),
):
    """
    Get timeline data showing number of CS papers by year.
    Returns data for 2015-2025 by default.
    """
    con = get_con()

    years = list(range(year_from, year_to + 1))

    data_df = con.execute(
        f"""
        SELECT 
            year,
            COUNT(*) as paper_count,
            SUM(Patent_Count) as patent_count
        FROM read_parquet('{PARQUET_DASHBOARD}')
        WHERE year >= ? AND year <= ?
        GROUP BY year
        ORDER BY year
    """,
        [year_from, year_to],
    ).fetchdf()

    timeline_data = []
    for year in years:
        year_data = data_df[data_df["year"] == year]
        if not year_data.empty:
            timeline_data.append(
                {
                    "year": year,
                    "paper_count": int(year_data["paper_count"].iloc[0]),
                    "patent_count": int(year_data["patent_count"].iloc[0]),
                }
            )
        else:
            timeline_data.append({"year": year, "paper_count": 0, "patent_count": 0})

    timeline_df = pd.DataFrame(timeline_data)

    return {
        "timeline": to_records(timeline_df),
        "meta": {
            "year_from": year_from,
            "year_to": year_to,
            "total_papers": (
                int(timeline_df["paper_count"].sum()) if not timeline_df.empty else 0
            ),
        },
    }


@app.get("/api/dashboard/patent-histogram")
def get_patent_histogram_data(
    year: Optional[int] = Query(None, ge=2015, le=2025),
    year_from: int = Query(2015, ge=2015, le=2025),
    year_to: int = Query(2025, ge=2015, le=2025),
):
    """
    Get patent citation distribution data.
    If year is specified, returns data for that specific year.
    Otherwise, returns data for the entire year range.
    """
    con = get_con()

    # Build WHERE clause
    where_conditions = ["year >= ?", "year <= ?"]
    params = [year_from, year_to]

    if year is not None:
        where_conditions = ["year = ?"]
        params = [year]

    where_clause = " AND ".join(where_conditions)

    histogram_df = con.execute(
        f"""
        SELECT 
            patent_count,
            COUNT(*) as frequency
        FROM read_parquet('{PARQUET_DASHBOARD}')
        WHERE {where_clause}
        GROUP BY patent_count
        ORDER BY patent_count
    """,
        params,
    ).fetchdf()

    if histogram_df.empty:
        histogram_df = pd.DataFrame(
            {
                "patent_count": [0],
                "frequency": [0],
            }
        )

    stats_df = con.execute(
        f"""
        SELECT 
            COUNT(*) as total_papers,
            AVG(patent_count) as mean_patent_count,
            MIN(patent_count) as min_patent_count,
            MAX(patent_count) as max_patent_count,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY patent_count) as median_patent_count
        FROM read_parquet('{PARQUET_DASHBOARD}')
        WHERE {where_clause}
    """,
        params,
    ).fetchdf()

    histogram_df = histogram_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    histogram_df["patent_count"] = histogram_df["patent_count"].astype(int)
    histogram_df["frequency"] = histogram_df["frequency"].astype(int)

    if not stats_df.empty:
        stats_df = stats_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        stats = {
            "total_papers": int(stats_df.loc[0, "total_papers"]),
            "mean_patent_count": float(stats_df.loc[0, "mean_patent_count"]),
            "min_patent_count": float(stats_df.loc[0, "min_patent_count"]),
            "max_patent_count": float(stats_df.loc[0, "max_patent_count"]),
            "median_patent_count": float(stats_df.loc[0, "median_patent_count"]),
        }
    else:
        stats = {
            "total_papers": 0,
            "mean_patent_count": 0.0,
            "min_patent_count": 0.0,
            "max_patent_count": 0.0,
            "median_patent_count": 0.0,
        }

    return {
        "histogram": to_records(histogram_df),
        "statistics": stats,
        "meta": {
            "year": year,
            "year_from": year_from,
            "year_to": year_to,
            "selected_year": year is not None,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
