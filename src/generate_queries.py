"""Generate the persona query dataset from Nemotron-Personas-Belgium with a local Ollama model.

    python src/generate_queries.py --language nl --count 4000

Personas are taken in a fixed pseudo-random order, so a larger --count extends an
earlier run. The run can be stopped at any time and resumed with the same command:
saved personas and rejected personas are skipped.

Output, one line per persona: data/queries/<language>.jsonl
Rejected personas and the reason:  data/queries/<language>.failures.jsonl
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from query_generator import QueryGenerator, profile_text

NEMOTRON = "nvidia/Nemotron-Personas-Belgium"
DATA_DIR = Path(__file__).parent.parent / "data"

# Personas used while developing the prompt; they are never part of the dataset.
DEVELOPMENT_UUIDS = {
    "4e42a5ef-c349-4543-8a5d-30bf7ca1f21c",
    "80e0f7f8-8759-4027-8a36-adb5da6f12c3",
    "d9f08750-fb65-4b19-80bb-6bccd2dfa484",
    "76702d7a-b339-4121-a8db-e3e246702280",
}


def nemotron_file(language: str) -> str:
    """Download (once) the Nemotron split of one language: 300,000 personas, about 1 GB."""
    return hf_hub_download(NEMOTRON, f"data/{language}_BE.parquet", repo_type="dataset",
                           local_dir=DATA_DIR / "nemotron")


def load_personas(path: str, uuids: list[str]) -> dict[str, dict]:
    """Read only the requested personas; the full file would take about 6 GB of RAM."""
    wanted, personas = set(uuids), {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        personas.update({p["uuid"]: p for p in batch.to_pylist() if p["uuid"] in wanted})
    return personas


def persona_order(uuids, seed: int) -> list[str]:
    """A fixed pseudo-random order, identical for both languages (they share UUIDs)."""
    candidates = set(uuids) - DEVELOPMENT_UUIDS
    return sorted(candidates, key=lambda uuid: hashlib.sha256(f"{seed}:{uuid}".encode()).digest())


def read_uuids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["uuid"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", choices=("nl", "en"), required=True)
    parser.add_argument("--count", type=int, default=10, help="Number of personas to process in total.")
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama model (default: qwen3.5:9b).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unload-every", type=int, default=64,
                        help="Free the Ollama model every N personas to keep memory use flat.")
    args = parser.parse_args()

    output = DATA_DIR / "queries" / f"{args.language}.jsonl"
    failures = output.with_suffix(".failures.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    done = read_uuids(output) | read_uuids(failures)

    path = nemotron_file(args.language)
    all_uuids = pq.read_table(path, columns=["uuid"]).column("uuid").to_pylist()
    todo = [uuid for uuid in persona_order(all_uuids, args.seed)[:args.count] if uuid not in done]
    personas = load_personas(path, todo)
    print(f"{len(done)} personas already processed, {len(todo)} to go.", flush=True)

    generator = QueryGenerator(args.model, args.language, args.seed)
    consecutive_failures = 0
    for i, uuid in enumerate(todo, 1):
        if i % args.unload_every == 0:
            generator.unload()
        persona, started = personas[uuid], time.monotonic()
        try:
            queries = generator.generate(persona)
        except (ValueError, RuntimeError, OSError) as error:
            append(failures, {"uuid": uuid, "language": args.language, "error": str(error)})
            print(f"[{i}/{len(todo)}] {uuid} rejected: {error}", flush=True)
            consecutive_failures += 1
            if consecutive_failures == 3:
                raise SystemExit("Stopping after three failures in a row. Is Ollama running?")
            continue
        consecutive_failures = 0
        append(output, {"uuid": uuid, "language": args.language,
                        "queries": [q["query"] for q in queries],
                        "source_fields": [q["source_fields"] for q in queries],
                        "profile": profile_text(persona), "model": args.model})
        print(f"[{i}/{len(todo)}] {uuid} saved in {time.monotonic() - started:.1f}s", flush=True)
    generator.unload()


if __name__ == "__main__":
    main()
