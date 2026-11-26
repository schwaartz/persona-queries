"""Convert the generated queries into the files of the Hugging Face dataset.

    python src/export_dataset.py

Reads data/queries/<language>.jsonl and writes data/hf/<language>_BE.jsonl (the split
names of Nemotron-Personas-Belgium), one row per persona:

    {"uuid": ..., "query_1": ..., "query_2": ..., "query_3": ...}

The profiles are left out: they are part of Nemotron-Personas-Belgium and can be
joined back on `uuid`. The published dataset is EdwardHuy/persona-queries.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

for language in ("nl", "en"):
    source = DATA_DIR / "queries" / f"{language}.jsonl"
    if not source.exists():
        print(f"{source} not found, skipped")
        continue
    target = DATA_DIR / "hf" / f"{language}_BE.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    with target.open("w", encoding="utf-8") as file:
        for row in rows:
            queries = row["queries"] + [None] * (3 - len(row["queries"]))
            flat = {"uuid": row["uuid"], **{f"query_{i}": q for i, q in enumerate(queries, 1)}}
            file.write(json.dumps(flat, ensure_ascii=False) + "\n")
    print(f"{target}: {len(rows)} personas")
