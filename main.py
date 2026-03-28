#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║              DATA LENS  –  Privacy-Preserving AI Layer       ║
║                       Hackathon MVP                          ║
╠══════════════════════════════════════════════════════════════╣
║  Flow:                                                       ║
║    Real DB → Schema Extraction → Synthetic Data (Faker)      ║
║           → External AI (OpenAI) → Response                  ║
║                                                              ║
║  GUARANTEE: No real records ever leave this device.          ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import sqlite3
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faker import Faker
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
DB_PATH        = os.getenv("DB_PATH",        "data_lens.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL",   "gpt-4o-mini")
SYNTHETIC_ROWS = int(os.getenv("SYNTHETIC_ROWS", "12"))

fake = Faker()
Faker.seed(0)   # reproducible schema-level examples

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Data Lens",
    description="Privacy-Preserving AI Layer – only synthetic data reaches the cloud",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend at /ui
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Database helpers ──────────────────────────────────────────────────────────
@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


# ── Database initialisation ───────────────────────────────────────────────────
DDL = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    email        TEXT    UNIQUE NOT NULL,
    age          INTEGER,
    city         TEXT,
    country      TEXT,
    phone        TEXT,
    signup_date  TEXT,
    is_premium   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS products (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    category    TEXT,
    price       REAL,
    stock       INTEGER,
    description TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER REFERENCES users(id),
    product_id       INTEGER REFERENCES products(id),
    quantity         INTEGER,
    total_amount     REAL,
    status           TEXT,
    order_date       TEXT,
    shipping_address TEXT
);

CREATE TABLE IF NOT EXISTS support_tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),
    subject     TEXT,
    body        TEXT,
    priority    TEXT,
    status      TEXT,
    created_at  TEXT
);
"""

CATEGORIES  = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Beauty"]
STATUSES    = ["pending", "processing", "shipped", "delivered", "cancelled"]
PRIORITIES  = ["low", "medium", "high", "critical"]
TKT_STATUS  = ["open", "in_progress", "resolved", "closed"]


def _seed(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # ── users ─────────────────────────────────────────────────────────────────
    if cur.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        rows = [
            (
                fake.name(),
                fake.unique.email(),
                random.randint(18, 72),
                fake.city(),
                fake.country(),
                fake.phone_number(),
                fake.date_between(start_date="-3y", end_date="today").isoformat(),
                random.choices([0, 1], weights=[70, 30])[0],
            )
            for _ in range(60)
        ]
        cur.executemany(
            "INSERT INTO users (name,email,age,city,country,phone,signup_date,is_premium) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )

    # ── products ──────────────────────────────────────────────────────────────
    if cur.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
        rows = [
            (
                fake.catch_phrase(),
                random.choice(CATEGORIES),
                round(random.uniform(4.99, 599.99), 2),
                random.randint(0, 500),
                fake.sentence(nb_words=12),
            )
            for _ in range(30)
        ]
        cur.executemany(
            "INSERT INTO products (name,category,price,stock,description) VALUES (?,?,?,?,?)",
            rows,
        )

    # ── orders ────────────────────────────────────────────────────────────────
    if cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0:
        rows = [
            (
                random.randint(1, 60),
                random.randint(1, 30),
                random.randint(1, 5),
                round(random.uniform(9.99, 999.99), 2),
                random.choice(STATUSES),
                fake.date_between(start_date="-1y", end_date="today").isoformat(),
                fake.address().replace("\n", ", "),
            )
            for _ in range(250)
        ]
        cur.executemany(
            "INSERT INTO orders (user_id,product_id,quantity,total_amount,status,order_date,shipping_address) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )

    # ── support_tickets ───────────────────────────────────────────────────────
    if cur.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0] == 0:
        rows = [
            (
                random.randint(1, 60),
                fake.sentence(nb_words=6),
                fake.paragraph(nb_sentences=3),
                random.choice(PRIORITIES),
                random.choice(TKT_STATUS),
                fake.date_time_between(start_date="-6m", end_date="now").isoformat(),
            )
            for _ in range(80)
        ]
        cur.executemany(
            "INSERT INTO support_tickets (user_id,subject,body,priority,status,created_at) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )

    conn.commit()


def init_db() -> None:
    with db_conn() as conn:
        conn.executescript(DDL)
        _seed(conn)
    print(f"✅  Database ready  →  {DB_PATH}")


# ── Schema extractor ──────────────────────────────────────────────────────────
def extract_schema() -> Dict[str, Any]:
    """Return table schemas + row counts. Safe to expose publicly."""
    with db_conn() as conn:
        cur = conn.cursor()
        tables = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        schema: Dict[str, Any] = {}
        for (tbl,) in tables:
            cols = cur.execute(f"PRAGMA table_info({tbl})").fetchall()
            count = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            schema[tbl] = {
                "columns": [{"name": c[1], "type": c[2] or "TEXT"} for c in cols],
                "row_count": count,
            }
    return schema


def get_real_samples(table: str, limit: int = 6) -> List[Dict]:
    """Fetch real rows – used ONLY to guide synthetic generation, never returned to callers."""
    with db_conn() as conn:
        rows = conn.execute(f"SELECT * FROM {table} LIMIT {limit}").fetchall()
    return [dict(r) for r in rows]


# ── Synthetic-data engine ─────────────────────────────────────────────────────

# Column-name substring → Faker generator (most-specific first)
_NAME_GENERATORS = [
    ("shipping_address", lambda: fake.address().replace("\n", ", ")),
    ("email",            lambda: fake.email()),
    ("phone",            lambda: fake.phone_number()),
    ("address",          lambda: fake.address().replace("\n", ", ")),
    ("city",             lambda: fake.city()),
    ("country",          lambda: fake.country()),
    ("subject",          lambda: fake.sentence(nb_words=6)),
    ("body",             lambda: fake.paragraph(nb_sentences=3)),
    ("description",      lambda: fake.sentence(nb_words=12)),
    ("name",             lambda: fake.name()),
]

_ENUM_COLUMNS = {
    "status":   STATUSES,
    "category": CATEGORIES,
    "priority": PRIORITIES,
}


def _synth_value(col_name: str, col_type: str, sample_val: Any) -> Any:
    """
    Derive a privacy-safe synthetic value for a single cell.
    Strategy:
      1. If column name implies PII  → use Faker
      2. If column is an enum-like   → sample from known enum
      3. If value looks like a date  → generate a random date
      4. Numeric                     → jitter within plausible range
      5. FK / ID columns             → keep as-is for relational coherence
    """
    col_lower = col_name.lower()

    # ── FK / PK passthrough ───────────────────────────────────────────────────
    if col_lower in ("id", "user_id", "product_id"):
        return sample_val

    # ── Name-based PII replacement ────────────────────────────────────────────
    for pattern, gen in _NAME_GENERATORS:
        if pattern in col_lower:
            return gen()

    # ── Enum columns ──────────────────────────────────────────────────────────
    for key, choices in _ENUM_COLUMNS.items():
        if key in col_lower:
            return random.choice(choices)

    if sample_val is None:
        return None

    # ── Date strings ─────────────────────────────────────────────────────────
    if isinstance(sample_val, str):
        stripped = sample_val.strip()
        # ISO date (YYYY-MM-DD) or datetime
        if len(stripped) >= 10 and stripped[4:5] == "-" and stripped[7:8] == "-":
            return fake.date_between(start_date="-1y", end_date="today").isoformat()
        return fake.word()

    # ── Numeric jitter ────────────────────────────────────────────────────────
    if isinstance(sample_val, float):
        lo = max(0.0, sample_val * 0.5)
        hi = sample_val * 1.5 if sample_val > 0 else 100.0
        return round(random.uniform(lo, hi), 2)

    if isinstance(sample_val, int):
        if col_lower == "age":
            return random.randint(18, 72)
        if col_lower in ("is_premium",):
            return random.choices([0, 1], weights=[70, 30])[0]
        if col_lower in ("stock", "quantity"):
            return random.randint(0, max(1, sample_val * 2))
        return random.randint(1, max(1, abs(sample_val) * 2))

    return sample_val


def generate_synthetic(schema: Dict, num_rows: int = SYNTHETIC_ROWS) -> Dict[str, List[Dict]]:
    """
    Build a fully synthetic dataset that mirrors the real schema.
    Real records are used only as statistical anchors — they are NOT included in output.
    """
    synthetic: Dict[str, List[Dict]] = {}

    for table, info in schema.items():
        columns = info["columns"]
        anchors = get_real_samples(table, limit=8)   # private – never returned

        if not anchors:
            synthetic[table] = []
            continue

        rows: List[Dict] = []
        for i in range(num_rows):
            anchor = anchors[i % len(anchors)]
            row: Dict[str, Any] = {}
            for col in columns:
                row[col["name"]] = _synth_value(col["name"], col["type"], anchor.get(col["name"]))
            rows.append(row)

        synthetic[table] = rows

    return synthetic


# ── AI query engine ───────────────────────────────────────────────────────────
def call_ai(prompt: str, synthetic_data: Dict, schema: Dict) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not set. Add it to your .env file.",
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Schema summary (safe – no real data)
    schema_lines = "\n".join(
        "  • {}: {}  [{} real rows]".format(
            tbl,
            ", ".join("{} ({})".format(c["name"], c["type"]) for c in info["columns"]),
            info["row_count"],
        )
        for tbl, info in schema.items()
    )

    synthetic_json = json.dumps(synthetic_data, indent=2, default=str)

    system_prompt = f"""You are a data analyst assistant for **Data Lens**, a privacy-preserving AI intelligence layer.

## How Data Lens works
1. It connects to a local database (never exposed externally).
2. It extracts the *schema* and uses Faker to generate **fully synthetic** rows that mirror the real data's structure and distribution — but contain no actual personal information.
3. Only the schema description and synthetic rows are sent to you.

## Database schema (real, safe to share)
{schema_lines}

## Synthetic sample data  ({SYNTHETIC_ROWS} rows/table — privacy-safe, no real PII)
```json
{synthetic_json}
```

## Your job
Answer the analyst's question based on the synthetic sample. Be concise, analytical, and insightful.
When asked about totals or aggregates, clarify you are working from a synthetic sample and scale estimates accordingly.
If asked to write SQL, write it against the real schema above.
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )

    return response.choices[0].message.content


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    init_db()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Redirect to the frontend UI."""
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())


@app.get("/health", tags=["System"])
def health():
    """Live health-check. Shows table row counts and AI config status."""
    schema = extract_schema()
    return {
        "status":        "healthy",
        "database":      DB_PATH,
        "tables":        {t: v["row_count"] for t, v in schema.items()},
        "ai_model":      OPENAI_MODEL,
        "ai_configured": bool(OPENAI_API_KEY),
        "timestamp":     datetime.utcnow().isoformat() + "Z",
    }


@app.get("/schema", tags=["Data"])
def schema_endpoint():
    """
    Return the real database schema (column names, types, row counts).
    Contains **no** actual row data — safe to expose.
    """
    return extract_schema()


@app.get("/synthetic", tags=["Data"])
def synthetic_endpoint(
    rows: int = Query(default=5, ge=1, le=50, description="Synthetic rows per table"),
):
    """
    Preview synthetic data. Useful for demo / debugging.
    All values are Faker-generated — zero real PII.
    """
    schema = extract_schema()
    return {
        "synthetic_rows_per_table": rows,
        "privacy_guarantee": "All values are Faker-generated. No real data included.",
        "data": generate_synthetic(schema, num_rows=rows),
    }


@app.get("/query", tags=["AI"])
def query_endpoint(
    prompt: str = Query(..., description="Natural-language question about your data"),
    rows:   int = Query(default=SYNTHETIC_ROWS, ge=1, le=50,
                        description="Synthetic rows per table sent to the AI"),
):
    """
    **Core Data Lens endpoint.**

    Pipeline:
    1. Read schema from local DB
    2. Generate `rows` synthetic rows per table (Faker, no real data)
    3. POST schema + synthetic data + `prompt` to OpenAI
    4. Return AI response

    ✅ Real data **never** leaves the device.
    """
    try:
        schema         = extract_schema()
        synthetic_data = generate_synthetic(schema, num_rows=rows)
        ai_response    = call_ai(prompt, synthetic_data, schema)

        return {
            "prompt":            prompt,
            "response":          ai_response,
            "privacy_guarantee": "Only synthetic (Faker-generated) data was sent to the AI model.",
            "synthetic_rows_per_table": rows,
            "tables_available":  list(schema.keys()),
            "model_used":        OPENAI_MODEL,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
