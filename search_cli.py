import os
import sys
import json
from pathlib import Path
import math


INDEX_FILE = Path(__file__).parent / "index" / "chunks.jsonl"


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def load_index() -> list[dict]:
    rows = []
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def embed_query(query: str) -> list[float]:
    from datapizzai.clients.openai_client import OpenAIClient

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY")
    client = OpenAIClient(api_key=api_key, model=model)
    return client.embed(query, model)


def main():
    if len(sys.argv) < 2:
        print("Usage: python search_cli.py 'your question'")
        sys.exit(1)
    query = sys.argv[1]
    rows = load_index()
    q = embed_query(query)
    scored = [(cosine(q, r.get("embedding", [])), r) for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    for s, r in scored[:5]:
        print(f"score={s:.3f} | {r['source']}")
        print(r["text"][:300].replace("\n", " ") + ("..." if len(r["text"]) > 300 else ""))
        print()


if __name__ == "__main__":
    main()


