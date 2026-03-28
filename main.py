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
import shutil
import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

import uuid as _uuid
import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faker import Faker
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# ── Settings persistence (settings.json) ────────────────────────────────────
SETTINGS_FILE = Path("settings.json")

_DEFAULT_FTS: Dict[str, Any] = {
    "enabled": False,
    "host":    "localhost",
    "port":    9200,
    "type":    "elasticsearch",
    "index":   "",
    "api_key": "",
}


def _load_settings() -> Dict[str, Any]:
    s: Dict[str, Any] = {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_model":   os.getenv("OPENAI_MODEL",   "gpt-4o-mini"),
        "synthetic_rows": int(os.getenv("SYNTHETIC_ROWS", "12")),
        "databases":      [],
        "folders":        [],
        "fts":            dict(_DEFAULT_FTS),
    }
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text())
            for k in s:
                if k in saved:
                    s[k] = saved[k]
        except Exception:
            pass
    # Seed default DB from env if no databases configured yet
    if not s["databases"]:
        s["databases"].append({
            "id":    "default",
            "name":  "default",
            "path":  os.getenv("DB_PATH", "data_lens.db"),
            "active": True,
        })
    return s


app_settings: Dict[str, Any] = _load_settings()


def _save_settings() -> None:
    SETTINGS_FILE.write_text(json.dumps(app_settings, indent=2))


def _api_key()    -> str:       return app_settings["openai_api_key"]
def _model()      -> str:       return app_settings["openai_model"]
def _syn_rows()   -> int:       return app_settings["synthetic_rows"]
def _active_dbs() -> List[Dict]: return [db for db in app_settings["databases"] if db.get("active")]
def _db_path()    -> str:
    dbs = _active_dbs()
    return dbs[0]["path"] if dbs else "data_lens.db"

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
def db_conn(path: Optional[str] = None):
    conn = sqlite3.connect(path or _db_path())
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


def init_db(path: Optional[str] = None) -> None:
    """Create & seed the DB only when it has no user tables yet."""
    target = path or _db_path()
    with db_conn(target) as conn:
        existing = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0]
        if existing == 0:
            conn.executescript(DDL)
            _seed(conn)
            print(f"✅  New database seeded  →  {target}")
        else:
            print(f"✅  Database connected  →  {target}  ({existing} existing tables)")


# ── Schema extractor ──────────────────────────────────────────────────────────
def extract_schema_from(path: str) -> Dict[str, Any]:
    """Extract table schemas + row counts from a specific SQLite file."""
    with db_conn(path) as conn:
        cur = conn.cursor()
        tables = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        schema: Dict[str, Any] = {}
        for (tbl,) in tables:
            cols  = cur.execute(f"PRAGMA table_info({tbl})").fetchall()
            count = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            schema[tbl] = {
                "columns":   [{"name": c[1], "type": c[2] or "TEXT"} for c in cols],
                "row_count": count,
            }
    return schema


def extract_schema() -> Dict[str, Any]:
    """Backward-compat: extract schema from the primary active DB."""
    return extract_schema_from(_db_path())


def extract_schemas_for(db_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Merge schemas from selected (or all active) databases.
    Keys are prefixed with '<db_name>.' when more than one DB is queried.
    Internal metadata (_db_id, _db_path, _table) is stored per-entry for
    later use in synthetic generation — never returned to callers directly.
    """
    if db_ids:
        dbs = [d for d in app_settings["databases"] if d["id"] in db_ids]
    else:
        dbs = _active_dbs()
    if not dbs:
        raise HTTPException(status_code=400, detail="No databases selected or configured.")
    multi  = len(dbs) > 1
    result: Dict[str, Any] = {}
    for db in dbs:
        try:
            schema = extract_schema_from(db["path"])
        except Exception:
            continue
        for tbl, info in schema.items():
            key = f"{db['name']}.{tbl}" if multi else tbl
            result[key] = {**info, "_db_id": db["id"], "_db_path": db["path"], "_table": tbl}
    return result


def get_real_samples(table: str, limit: int = 6, db_path: Optional[str] = None) -> List[Dict]:
    """Fetch real rows – used ONLY to guide synthetic generation, never returned to callers."""
    with db_conn(db_path or _db_path()) as conn:
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


def generate_synthetic(schema: Dict, num_rows: int = 12) -> Dict[str, List[Dict]]:
    """
    Build a fully synthetic dataset that mirrors the real schema.
    Real records are used only as statistical anchors — they are NOT included in output.
    Supports multi-db schemas via _db_path / _table metadata injected by extract_schemas_for.
    """
    synthetic: Dict[str, List[Dict]] = {}

    for table_key, info in schema.items():
        columns  = info["columns"]
        db_path  = info.get("_db_path")
        raw_tbl  = info.get("_table", table_key)
        anchors  = get_real_samples(raw_tbl, limit=8, db_path=db_path)   # private – never returned

        if not anchors:
            synthetic[table_key] = []
            continue

        rows: List[Dict] = []
        for i in range(num_rows):
            anchor = anchors[i % len(anchors)]
            row: Dict[str, Any] = {}
            for col in columns:
                row[col["name"]] = _synth_value(col["name"], col["type"], anchor.get(col["name"]))
            rows.append(row)

        synthetic[table_key] = rows

    return synthetic


def _clean_schema(schema: Dict) -> Dict:
    """Strip internal metadata keys (prefixed _) before sending to client or AI."""
    return {k: {ik: iv for ik, iv in v.items() if not ik.startswith("_")} for k, v in schema.items()}


# ── AI query engine ───────────────────────────────────────────────────────────
def call_ai(prompt: str, synthetic_data: Dict, schema: Dict) -> str:
    if not _api_key():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not set. Configure it via the ⚙ Settings panel or .env file.",
        )

    client = OpenAI(api_key=_api_key())

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

## Synthetic sample data  ({_syn_rows()} rows/table — privacy-safe, no real PII)
```json
{synthetic_json}
```

## Your job
Answer the analyst's question based on the synthetic sample. Be concise, analytical, and insightful.
When asked about totals or aggregates, clarify you are working from a synthetic sample and scale estimates accordingly.
If asked to write SQL, write it against the real schema above.
"""

    response = client.chat.completions.create(
        model=_model(),
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
    """Live health-check. Shows all active databases and AI config status."""
    active = _active_dbs()
    db_info = {}
    for db in active:
        try:
            s = extract_schema_from(db["path"])
            db_info[db["name"]] = {"path": db["path"], "tables": {t: v["row_count"] for t, v in s.items()}}
        except Exception as e:
            db_info[db["name"]] = {"path": db["path"], "error": str(e)}
    return {
        "status":        "healthy",
        "databases":     db_info,
        "ai_model":      _model(),
        "ai_configured": bool(_api_key()),
        "timestamp":     datetime.utcnow().isoformat() + "Z",
    }


@app.get("/schema", tags=["Data"])
def schema_endpoint(
    db_ids: Optional[str] = Query(None, description="Comma-separated database IDs (default: all active)"),
):
    """
    Return table schemas (column names, types, row counts) from selected databases.
    Contains **no** actual row data — safe to expose.
    """
    ids    = [x.strip() for x in db_ids.split(",")] if db_ids else None
    schema = extract_schemas_for(ids)
    return _clean_schema(schema)


@app.get("/synthetic", tags=["Data"])
def synthetic_endpoint(
    rows:   int            = Query(default=5, ge=1, le=50, description="Synthetic rows per table"),
    db_ids: Optional[str]  = Query(None, description="Comma-separated database IDs"),
):
    """
    Preview synthetic data. Useful for demo / debugging.
    All values are Faker-generated — zero real PII.
    """
    ids    = [x.strip() for x in db_ids.split(",")] if db_ids else None
    schema = extract_schemas_for(ids)
    return {
        "synthetic_rows_per_table": rows,
        "privacy_guarantee": "All values are Faker-generated. No real data included.",
        "data": generate_synthetic(schema, num_rows=rows),
    }


@app.get("/query", tags=["AI"])
def query_endpoint(
    prompt:  str           = Query(...,  description="Natural-language question about your data"),
    rows:    Optional[int] = Query(None, ge=1, le=50,
                                   description="Synthetic rows per table (defaults to SYNTHETIC_ROWS in config)"),
    db_ids:  Optional[str] = Query(None, description="Comma-separated database IDs to query (default: all active)"),
):
    """
    **Core Data Lens endpoint.**

    Pipeline:
    1. Read schema from selected (or all active) databases
    2. Generate `rows` synthetic rows per table (Faker, no real data)
    3. POST schema + synthetic data + `prompt` to OpenAI
    4. Return AI response

    ✅ Real data **never** leaves the device.
    """
    try:
        id_list        = [x.strip() for x in db_ids.split(",")] if db_ids else None
        n_rows         = rows if rows is not None else _syn_rows()
        schema         = extract_schemas_for(id_list)
        synthetic_data = generate_synthetic(schema, num_rows=n_rows)
        clean          = _clean_schema(schema)
        ai_response    = call_ai(prompt, synthetic_data, clean)
        queried_dbs    = (
            [d["name"] for d in app_settings["databases"] if d["id"] in id_list]
            if id_list else [d["name"] for d in _active_dbs()]
        )
        return {
            "prompt":                   prompt,
            "response":                 ai_response,
            "privacy_guarantee":        "Only synthetic (Faker-generated) data was sent to the AI model.",
            "synthetic_rows_per_table": n_rows,
            "tables_available":         list(clean.keys()),
            "model_used":               _model(),
            "databases_queried":        queried_dbs,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Config endpoints ──────────────────────────────────────────────────────────
class ConfigUpdate(BaseModel):
    db_path:        Optional[str] = None
    openai_api_key: Optional[str] = None   # send empty string "" to clear
    openai_model:   Optional[str] = None
    synthetic_rows: Optional[int] = None


@app.get("/config", tags=["System"])
def get_config():
    """Return current runtime configuration (API key masked)."""
    key = _api_key()
    return {
        "db_path":             _db_path(),
        "openai_model":        _model(),
        "synthetic_rows":      _syn_rows(),
        "openai_api_key_set":  bool(key),
        "openai_api_key_hint": (key[:4] + "…" + key[-4:]) if len(key) > 8 else ("(not set)" if not key else "***"),
    }


@app.patch("/config", tags=["System"])
def update_config(body: ConfigUpdate):
    """Update one or more config values at runtime and persist to settings.json."""
    db_changed = False
    if body.db_path is not None and body.db_path != _db_path():
        # Update the first active db or add a new entry
        dbs = app_settings["databases"]
        if dbs:
            dbs[0]["path"] = body.db_path
        else:
            dbs.append({"id": "default", "name": "default", "path": body.db_path, "active": True})
        db_changed = True
    if body.openai_api_key is not None:
        app_settings["openai_api_key"] = body.openai_api_key
    if body.openai_model is not None:
        app_settings["openai_model"] = body.openai_model
    if body.synthetic_rows is not None:
        app_settings["synthetic_rows"] = max(1, min(50, body.synthetic_rows))

    _save_settings()

    if db_changed:
        try:
            init_db(_db_path())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"DB switch failed: {exc}") from exc

    return {"ok": True, "config": get_config()}


@app.get("/db/test", tags=["Data"])
def test_db_connection(
    path: str = Query(..., description="Absolute or relative path to a SQLite .db file"),
):
    """Test a SQLite file path without changing the active database."""
    try:
        conn   = sqlite3.connect(path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        counts = {
            t[0]: conn.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
            for t in tables
        }
        conn.close()
        return {"ok": True, "path": path, "tables": counts, "table_count": len(tables)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/db/upload", tags=["Data"])
async def upload_db(file: UploadFile = File(...)):
    """
    Upload a SQLite file and switch the active database to it.
    The file is saved to the working directory; real data stays local.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".db", ".sqlite", ".sqlite3", ".s3db"):
        raise HTTPException(
            status_code=400,
            detail="Only .db / .sqlite / .sqlite3 / .s3db files are accepted.",
        )
    dest = Path(file.filename)
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    # Quick integrity check
    try:
        conn   = sqlite3.connect(str(dest))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        counts = {t[0]: conn.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0] for t in tables}
        conn.close()
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid SQLite file: {exc}") from exc

    # Register in app_settings databases list
    existing = next((d for d in app_settings["databases"] if d["path"] == str(dest)), None)
    if existing:
        existing["active"] = True
    else:
        app_settings["databases"].append({
            "id":          str(_uuid.uuid4())[:8],
            "name":        dest.stem,
            "path":        str(dest),
            "active":      True,
            "table_count": len(tables),
        })
    _save_settings()
    return {
        "ok":      True,
        "path":    str(dest),
        "tables":  counts,
        "message": f"Uploaded and added '{dest}' to your databases",
    }


@app.get("/db/discover", tags=["Data"])
def discover_databases():
    """List SQLite database files (.db / .sqlite) found in the working directory."""
    exts  = {".db", ".sqlite", ".sqlite3", ".s3db"}
    found = []
    cwd   = Path(".")
    for p in sorted(cwd.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                conn   = sqlite3.connect(str(p))
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                conn.close()
                found.append({
                    "path":        str(p),
                    "size_kb":     round(p.stat().st_size / 1024, 1),
                    "table_count": len(tables),
                    "active":      p.name == Path(_db_path()).name,
                })
            except Exception:
                pass
    return {"databases": found, "cwd": str(cwd.resolve())}


# ── Settings endpoints ────────────────────────────────────────────────────────

class DBAdd(BaseModel):
    path: str
    name: Optional[str] = None


class DBUpdate(BaseModel):
    name:   Optional[str]  = None
    active: Optional[bool] = None


class FTSConfig(BaseModel):
    enabled: Optional[bool] = None
    host:    Optional[str]  = None
    port:    Optional[int]  = None
    type:    Optional[str]  = None
    index:   Optional[str]  = None
    api_key: Optional[str]  = None


class FolderAdd(BaseModel):
    path: str


class AIConfig(BaseModel):
    openai_api_key: Optional[str] = None
    openai_model:   Optional[str] = None
    synthetic_rows: Optional[int] = None


@app.get("/settings", tags=["Settings"])
def get_settings():
    """Return all settings (databases, folders, FTS, AI config)."""
    key = _api_key()
    return {
        "databases":           app_settings["databases"],
        "folders":             app_settings["folders"],
        "fts":                 app_settings["fts"],
        "openai_model":        _model(),
        "synthetic_rows":      _syn_rows(),
        "openai_api_key_set":  bool(key),
        "openai_api_key_hint": (key[:4] + "…" + key[-4:]) if len(key) > 8 else ("(not set)" if not key else "***"),
    }


@app.patch("/settings/ai", tags=["Settings"])
def update_ai_settings(body: AIConfig):
    """Update AI / synthetic-data settings."""
    if body.openai_api_key is not None:
        app_settings["openai_api_key"] = body.openai_api_key
    if body.openai_model is not None:
        app_settings["openai_model"] = body.openai_model
    if body.synthetic_rows is not None:
        app_settings["synthetic_rows"] = max(1, min(50, body.synthetic_rows))
    _save_settings()
    return {"ok": True}


# ── Database management ───────────────────────────────────────────────────────

@app.get("/settings/databases", tags=["Settings"])
def list_settings_databases():
    """List all configured databases with their metadata."""
    result = []
    for db in app_settings["databases"]:
        entry = dict(db)
        try:
            s = extract_schema_from(db["path"])
            entry["table_count"] = len(s)
            entry["row_count"]   = sum(v["row_count"] for v in s.values())
            entry["reachable"]   = True
        except Exception:
            entry["reachable"] = False
        result.append(entry)
    return {"databases": result}


@app.post("/settings/databases", tags=["Settings"])
def add_settings_database(body: DBAdd):
    """Add a SQLite database to the configured list."""
    path = body.path.strip()
    try:
        conn   = sqlite3.connect(path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        counts = {t[0]: conn.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0] for t in tables}
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open database: {exc}") from exc
    if any(d["path"] == path for d in app_settings["databases"]):
        raise HTTPException(status_code=409, detail="Database already configured.")
    entry: Dict[str, Any] = {
        "id":          str(_uuid.uuid4())[:8],
        "name":        body.name or Path(path).stem,
        "path":        path,
        "active":      True,
        "table_count": len(tables),
    }
    app_settings["databases"].append(entry)
    _save_settings()
    return {"ok": True, "database": entry, "tables": counts}


@app.patch("/settings/databases/{db_id}", tags=["Settings"])
def update_settings_database(db_id: str, body: DBUpdate):
    """Rename or toggle a database's active state."""
    db = next((d for d in app_settings["databases"] if d["id"] == db_id), None)
    if not db:
        raise HTTPException(status_code=404, detail="Database not found.")
    if body.name is not None:
        db["name"] = body.name.strip() or db["name"]
    if body.active is not None:
        db["active"] = body.active
    _save_settings()
    return {"ok": True, "database": db}


@app.delete("/settings/databases/{db_id}", tags=["Settings"])
def delete_settings_database(db_id: str):
    """Remove a database from the configured list (file is NOT deleted)."""
    before = len(app_settings["databases"])
    app_settings["databases"] = [d for d in app_settings["databases"] if d["id"] != db_id]
    if len(app_settings["databases"]) == before:
        raise HTTPException(status_code=404, detail="Database not found.")
    _save_settings()
    return {"ok": True}


# ── FTS management ────────────────────────────────────────────────────────────

@app.get("/settings/fts", tags=["Settings"])
def get_fts_settings():
    """Return current Full-Text Search configuration."""
    return app_settings["fts"]


@app.patch("/settings/fts", tags=["Settings"])
def update_fts_settings(body: FTSConfig):
    """Update Full-Text Search configuration."""
    for field in ("enabled", "host", "port", "type", "index", "api_key"):
        val = getattr(body, field)
        if val is not None:
            app_settings["fts"][field] = val
    _save_settings()
    return {"ok": True, "fts": app_settings["fts"]}


# ── Folder management ─────────────────────────────────────────────────────────

@app.get("/settings/folders", tags=["Settings"])
def get_settings_folders():
    """List configured folder paths."""
    return {"folders": app_settings["folders"]}


@app.post("/settings/folders", tags=["Settings"])
def add_settings_folder(body: FolderAdd):
    """Add a folder to be scanned for SQLite files."""
    path = body.path.strip()
    if not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid directory: {path}")
    if path in app_settings["folders"]:
        raise HTTPException(status_code=409, detail="Folder already added.")
    exts = {".db", ".sqlite", ".sqlite3", ".s3db"}
    found_dbs = [str(p) for p in sorted(Path(path).iterdir())
                 if p.is_file() and p.suffix.lower() in exts]
    app_settings["folders"].append(path)
    _save_settings()
    return {"ok": True, "path": path, "found_dbs": found_dbs}


@app.delete("/settings/folders/{idx}", tags=["Settings"])
def remove_settings_folder(idx: int):
    """Remove a folder by its list index."""
    folders = app_settings["folders"]
    if idx < 0 or idx >= len(folders):
        raise HTTPException(status_code=404, detail="Folder index out of range.")
    removed = folders.pop(idx)
    _save_settings()
    return {"ok": True, "removed": removed}


@app.get("/settings/folders/scan", tags=["Settings"])
def scan_settings_folders():
    """Scan all configured folders for SQLite files."""
    exts   = {".db", ".sqlite", ".sqlite3", ".s3db"}
    result = []
    for folder in app_settings["folders"]:
        fp = Path(folder)
        if not fp.is_dir():
            continue
        for p in sorted(fp.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                result.append({
                    "path":     str(p),
                    "name":     p.stem,
                    "size_kb":  round(p.stat().st_size / 1024, 1),
                    "folder":   folder,
                    "imported": any(d["path"] == str(p) for d in app_settings["databases"]),
                })
    return {"files": result}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

