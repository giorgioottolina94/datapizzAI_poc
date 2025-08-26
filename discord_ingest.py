import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any

from datapizzai.clients.openai_client import OpenAIClient
from datapizzai.embedders.client_embedder import NodeEmbedder
from datapizzai.type.type import Chunk
from demo_minimo.components.text_splitter import TextSplitter
from demo_minimo.handbook_assistant.ingest import INDEX_FILE


DISCORD_API = "https://discord.com/api/v10"


def _api_get_messages(bot_token: str, channel_id: str, before: str | None = None, limit: int = 100) -> list[dict]:
    params = f"?limit={min(max(limit, 1), 100)}"
    if before:
        params += f"&before={before}"
    url = f"{DISCORD_API}/channels/{channel_id}/messages{params}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bot {bot_token}")
    req.add_header("User-Agent", "handbook-assistant (datapizzai-demo, 0.1)")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            retry_after = float(e.headers.get("Retry-After", "1"))
            time.sleep(min(retry_after, 5))
            return _api_get_messages(bot_token, channel_id, before, limit)
        elif e.code == 403:
            raise RuntimeError(
                f"❌ Accesso negato al canale {channel_id}.\n"
                "Possibili cause:\n"
                "• Bot token non valido o scaduto\n"
                "• Bot non ha accesso al server/canale\n"
                "• Mancano i permessi 'View Channel' e 'Read Message History'\n"
                "• Il bot non è stato invitato nel server"
            )
        elif e.code == 401:
            raise RuntimeError("❌ Bot token non valido. Verifica che sia corretto e non scaduto.")
        elif e.code == 404:
            raise RuntimeError(f"❌ Canale {channel_id} non trovato. Verifica che l'ID sia corretto.")
        else:
            raise RuntimeError(f"❌ Errore Discord API {e.code}: {e.reason}")


def backfill_channel(bot_token: str, channel_id: str, max_messages: int = 500) -> list[dict]:
    collected: list[dict] = []
    before: str | None = None
    while len(collected) < max_messages:
        batch = _api_get_messages(bot_token, channel_id, before=before, limit=100)
        if not batch:
            break
        collected.extend(batch)
        if len(batch) < 100:
            break
        before = batch[-1]["id"]
        # Gentle pacing to respect rate limits
        time.sleep(0.25)
    # Newest first from API; reverse to oldest-first for deterministic processing
    collected.reverse()
    return collected[:max_messages]


def _message_to_text(m: dict) -> str:
    author = m.get("author", {})
    author_name = author.get("username") or str(author.get("id") or "unknown")
    ts = m.get("timestamp", "")
    content = m.get("content") or ""
    attachments = m.get("attachments") or []
    embeds = m.get("embeds") or []
    parts: list[str] = [f"{author_name} [{ts}]\n{content}"]
    if attachments:
        parts.append("Allegati:\n" + "\n".join([a.get("url", "") for a in attachments if a.get("url")]))
    if embeds:
        descs = []
        for e in embeds:
            if e.get("title"):
                descs.append(str(e.get("title")))
            if e.get("description"):
                descs.append(str(e.get("description")))
        if descs:
            parts.append("Embeds:\n" + "\n".join(descs))
    return "\n\n".join([p for p in parts if p])


def _append_rows_to_index(rows: List[Dict[str, Any]]):
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: list[str] = []
    if INDEX_FILE.exists():
        existing_lines = INDEX_FILE.read_text(encoding="utf-8").splitlines()
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        for line in existing_lines:
            if line.strip():
                f.write(line + "\n")
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ingest_discord(bot_token: str, channel_ids: list[str], max_messages_per_channel: int = 500, batch_size: int = 64):
    if not bot_token or not channel_ids:
        raise RuntimeError("Token e lista canali sono obbligatori")

    # Ensure OpenAI API key is loaded from file if needed
    from demo_minimo.handbook_assistant.api.services import ensure_openai_key_env
    ensure_openai_key_env()
    
    api_key = os.getenv("OPENAI_API_KEY")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not api_key:
        raise RuntimeError("OpenAI API key non trovata. Configurala nel tab Config del sito web.")

    client = OpenAIClient(api_key=api_key, model=embed_model)
    splitter = TextSplitter(max_char=1500, overlap=150)
    embedder = NodeEmbedder(client=client, model_name=embed_model, batch_size=batch_size)

    all_chunks: list[Chunk] = []
    for ch in channel_ids:
        ch = str(ch).strip()
        if not ch:
            continue
        msgs = backfill_channel(bot_token, ch, max_messages=max_messages_per_channel)
        for m in msgs:
            text = _message_to_text(m)
            if not text.strip():
                continue
            chunks = splitter._run(text)
            message_id = m.get("id")
            timestamp = m.get("timestamp")
            author = (m.get("author") or {}).get("username") or str((m.get("author") or {}).get("id") or "")
            for c in chunks:
                c.metadata.update({
                    "source": f"discord://channel/{ch}/message/{message_id}",
                    "channel_id": ch,
                    "message_id": message_id,
                    "author": author,
                    "timestamp": timestamp,
                    "kind": "discord_message",
                })
            all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    all_chunks = embedder._run(all_chunks)

    # Prepare rows compatible with existing index schema
    out_rows: list[dict] = []
    for c in all_chunks:
        out_rows.append({
            "id": c.id,
            "text": c.text,
            "metadata": c.metadata,
            "embedding": c.embeddings[0].vector if c.embeddings else [],
        })
    _append_rows_to_index(out_rows)
    return len(out_rows)


