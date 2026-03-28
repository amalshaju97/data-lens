# 🔮 Data Lens — Privacy-Preserving AI Layer

> **Hackathon MVP** · Query your internal data through AI without ever exposing real records to the cloud.

---

## The Concept

```
Local DB  ──►  Schema Extract  ──►  Faker Synthetic Data  ──►  OpenAI  ──►  Response
  (real)          (safe)                  (safe)               (cloud)      (returned)
       ◄────────────────────────── REAL DATA NEVER CROSSES THIS LINE ──────────────────
```

Data Lens sits between your sensitive local database and powerful cloud AI models.  
It **reads your schema**, generates **fully synthetic rows** that mirror the real data's structure, and sends only those fake rows to the AI — so GPT never sees a single real name, email, or transaction.

---

## Quick Start

### 1 · Clone and enter the folder

```bash
cd /path/to/data-lens
```

### 2 · Create a virtual environment and install dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3 · Configure your OpenAI key

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

### 4 · Run

```bash
python main.py
# or with auto-reload for development:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5 · Open the UI

```
http://localhost:8000
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat frontend |
| `GET` | `/health` | Server + DB + AI status |
| `GET` | `/schema` | Real DB schema (no row data) |
| `GET` | `/synthetic?rows=N` | Preview synthetic data (N rows/table) |
| `GET` | `/query?prompt=...&rows=N` | ⭐ Main endpoint — NL query → AI response |
| `GET` | `/docs` | Swagger UI |

### Example query

```bash
curl "http://localhost:8000/query?prompt=What+are+the+top+product+categories+by+order+volume%3F"
```

```json
{
  "prompt": "What are the top product categories by order volume?",
  "response": "Based on the synthetic sample, Electronics and Clothing lead...",
  "privacy_guarantee": "Only synthetic (Faker-generated) data was sent to the AI model.",
  "synthetic_rows_per_table": 12,
  "tables_available": ["orders", "products", "support_tickets", "users"],
  "model_used": "gpt-4o-mini"
}
```

---

## Database Schema

The SQLite database is auto-seeded on first run with realistic-looking data:

| Table | Description | Rows |
|-------|-------------|------|
| `users` | Customers with name, email, age, city, country, phone, premium flag | 60 |
| `products` | Catalogue with name, category, price, stock | 30 |
| `orders` | Purchases linking users → products, with status and shipping | 250 |
| `support_tickets` | Customer issues with subject, body, priority, status | 80 |

---

## Privacy Guarantee

| What stays local | What goes to the cloud |
|-----------------|----------------------|
| All real DB rows | Only the schema (column names + types) |
| PII (names, emails, phones) | Faker-generated synthetic rows |
| Business-sensitive values | Numeric jitter of real magnitudes |
| Raw SQL data | AI-generated text response |

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI secret key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI chat model |
| `SYNTHETIC_ROWS` | `12` | Rows per table sent to AI |
| `DB_PATH` | `data_lens.db` | Path to the SQLite file |

---

## Tech Stack

- **Python 3.10+** — runtime
- **FastAPI** — REST API + auto-docs
- **Uvicorn** — ASGI server
- **SQLite** (stdlib) — local database
- **Faker** — synthetic data generation
- **OpenAI Python SDK** — AI queries
- **Vanilla HTML/CSS/JS** — zero-build frontend

---

## Project Structure

```
data-lens/
├── main.py          # Single-file FastAPI application
├── requirements.txt
├── .env.example
├── .env             # (gitignored — add your key here)
├── data_lens.db     # Auto-created SQLite DB
└── static/
    └── index.html   # Chat frontend
```

---

## Extending for Production

- **Swap SQLite → PostgreSQL / MongoDB** — change `db_conn()` to use `psycopg2` or `pymongo`
- **Add table/column access-controls** — role-based schema filtering before synthetic gen
- **On-device LLM** — replace `call_ai()` with a `GPT4All` / `Ollama` call for fully offline mode
- **Audit logging** — log every query, prompt, and synthetic payload for compliance
- **Differential privacy** — add Laplace noise to numeric columns before synthesis
