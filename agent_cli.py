import os
import json
from pathlib import Path

from datapizzai.agents import Agent
from datapizzai.clients.openai_client import OpenAIClient
from datapizzai.tools.tools import tool


INDEX_FILE = Path(__file__).parent / "index" / "chunks.jsonl"


def load_index():
    rows = []
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def cosine(a: list[float], b: list[float]) -> float:
    import math

    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def embed_query(query: str) -> list[float]:
    from datapizzai.clients.openai_client import OpenAIClient

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY")
    client = OpenAIClient(api_key=api_key, model=model)
    return client.embed(query, model)


@tool(name="kb_search", description="Cerca nei documenti indicizzati e restituisce i migliori passaggi con citazioni")
def kb_search(query: str, k: int = 5, mode: str | None = None) -> str:
    rows = load_index()
    q_lower = query.lower()
    if mode == "files" or any(t in q_lower for t in ["notebook", "ipynb", "esercizi", "exercise", ".ipynb"]):
        ipynb_rows = [r for r in rows if str(r.get("metadata", {}).get("source") or r.get("source", "")).endswith(".ipynb")]
        out = []
        for r in ipynb_rows[: max(k, 10)]:
            out.append({
                "source": r.get("metadata", {}).get("source") or r.get("source"),
                "kind": "ipynb",
            })
        return json.dumps(out, ensure_ascii=False)

    q = embed_query(query)
    scored = [(cosine(q, r.get("embedding", [])), r) for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, r in scored[:k]:
        out.append({
            "source": r.get("metadata", {}).get("source") or r.get("source"),
            "score": s,
            "text": r.get("text", "")[:400],
        })
    payload = json.dumps(out, ensure_ascii=False)
    return payload[:6000]


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY")

    system_prompt = (
        "Sei un assistant per knowledge base. Quando rispondi, chiama lo strumento kb_search"
        " per recuperare contesto e cita sempre le fonti. Se la risposta non è nei documenti, dillo chiaramente."
    )

    client = OpenAIClient(api_key=api_key, model=model)
    agent = Agent(
        name="hb-agent",
        system_prompt=system_prompt,
        client=client,
        tools=[kb_search],
        terminate_on_text=True,
    )

    print("=== Handbook Assistant ===")
    print("Type 'exit' to quit.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nBye!")
            break
        if q.lower() in {"exit", "quit", "bye", ""}:
            print("Bye!")
            break
        res = agent.run(q, tool_choice="required_first")
        print("Agent:", res)


if __name__ == "__main__":
    main()


