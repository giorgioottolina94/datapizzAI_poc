import json
import os
import sys
import subprocess
import importlib.util
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import urllib.parse
import urllib.request
import threading
import socket
from functools import partial

from datapizzai.agents import Agent
from datapizzai.clients.openai_client import OpenAIClient
from datapizzai.tools.tools import tool
from datapizzai.memory import Memory
from datapizzai.type.type import TextBlock, ROLE
from demo_minimo.handbook_assistant.discord_ingest import ingest_discord


INDEX_FILE = Path(__file__).parent / "index" / "chunks.jsonl"
STATE_DIR = Path(__file__).parent / "state"
HISTORY_FILE = STATE_DIR / "history.json"
PREVIEW_FILE = STATE_DIR / "preview_state.json"
REPORTS_DIR = Path(__file__).parent / "reports"
ROOT_DIR = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False, ttl=60)
def load_index():
    rows = []
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def cosine(a, b):
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
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    client = st.session_state["embed_client"]
    return client.embed(query, model)


@tool(name="kb_search", description="Cerca nei documenti indicizzati e restituisce passaggi con citazioni")
def kb_search(query: str, k: int = 5, mode: str | None = None) -> str:
    rows = st.session_state.get("index_rows", [])

    # Heuristic: per query su notebook/file, evita embedding full-scan e fai filename search
    q_lower = query.lower()
    if mode == "files" or any(t in q_lower for t in ["file", "notebook", "ipynb", "esercizi", "exercise", ".ipynb", "slides", "slide", "presentazione", "pdf", ".pdf", ".pptx", "python", ".py"]):
        week, day = _parse_time_filters(q_lower)
        allowed_exts = _infer_ext_filter(q_lower)

        candidates = rows
        if week or day:
            candidates = _pre_filter_by_path(candidates, week=week, day=day)
        candidates = _pre_filter_by_ext(candidates, allowed_exts)

        # basic name scoring
        def score_name(r):
            name = str(r.get("metadata", {}).get("source") or r.get("source", "")).lower()
            score = 0
            for tok in ["exercise", "eserc", "slides", "slide", "presentazione"]:
                if tok in name:
                    score += 1
            if q_lower in name:
                score += 1
            return score

        # dedupe by source
        seen = set()
        dedup = []
        for r in candidates:
            src = str(r.get("metadata", {}).get("source") or r.get("source", ""))
            if src in seen:
                continue
            seen.add(src)
            dedup.append(r)

        ranked = sorted(
            dedup,
            key=lambda r: (score_name(r), len(str(r.get("metadata", {}).get("source") or r.get("source", "")))),
            reverse=True,
        )
        out = []
        for r in ranked[: max(k, 10)]:
            src = r.get("metadata", {}).get("source") or r.get("source")
            kind = Path(src).suffix.lower().lstrip(".")
            out.append({
                "source": src,
                "kind": kind,
            })
        return json.dumps(out, ensure_ascii=False)

    # Default: dense embedding search
    q = embed_query(query)
    scored = [(cosine(q, r.get("embedding", [])), r) for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    seen_src = set()
    for s, r in scored:
        src = r.get("metadata", {}).get("source") or r.get("source")
        if src in seen_src:
            continue
        seen_src.add(src)
        out.append({
            "source": src,
            "score": s,
            "text": r.get("text", "")[:400],
        })
        if len(out) >= k:
            break
    payload = json.dumps(out, ensure_ascii=False)
    # Hard cap to avoid huge tool outputs in messages
    return payload[:6000]


@tool(name="nb_list", description="Elenca notebook .ipynb nelle cartelle locali. Restituisce JSON con percorsi.")
def nb_list(query: str = "", root: str = "auto", limit: int = 200) -> str:
    from demo_minimo.handbook_assistant.ingest import DATA_DIR, REMOTE_CACHE_DIR
    roots: list[Path]
    if root == "data":
        roots = [DATA_DIR]
    elif root == "remote":
        roots = [REMOTE_CACHE_DIR]
    else:
        roots = [REMOTE_CACHE_DIR, DATA_DIR]

    q = (query or "").lower().strip()
    results: list[str] = []
    for base in roots:
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
            for fn in filenames:
                if not fn.lower().endswith(".ipynb"):
                    continue
                p = str(Path(dirpath) / fn)
                if q and q not in p.lower():
                    continue
                results.append(p)
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break
    return json.dumps(results, ensure_ascii=False)


@tool(name="nb_run", description="Esegue un notebook e genera un report HTML. Accetta params_json opzionale.")
def nb_run(path: str, params_json: str | None = None, timeout_s: int = 900) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ip = Path(path)
    if not ip.exists() or ip.suffix.lower() != ".ipynb":
        return json.dumps({"ok": False, "error": "Notebook non trovato"}, ensure_ascii=False)
    # Parse params
    params: dict
    try:
        params = json.loads(params_json) if params_json else {}
        if not isinstance(params, dict):
            raise ValueError("params_json deve essere un oggetto JSON")
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Parametri non validi: {e}"}, ensure_ascii=False)

    # Build output paths
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", ip.stem)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_ipynb = REPORTS_DIR / f"{safe_name}__{ts}.executed.ipynb"
    out_html = REPORTS_DIR / f"{safe_name}__{ts}.report.html"

    # Execute with papermill
    try:
        import papermill as pm
        # Protegge la working directory di Streamlit
        _orig_cwd = os.getcwd()
        try:
            pm.execute_notebook(
                str(ip),
                str(out_ipynb),
                parameters=params,
                cwd=str(ip.parent),
                kernel_name="python3",
                progress_bar=False,
                request_save_on_cell_execute=True,
            )
        finally:
            try:
                os.chdir(_orig_cwd)
            except Exception:
                pass
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Errore esecuzione: {e}"}, ensure_ascii=False)

    # Convert to HTML
    try:
        import nbformat
        from nbconvert import HTMLExporter
        nb = nbformat.read(str(out_ipynb), as_version=4)
        exporter = HTMLExporter()
        body, _ = exporter.from_notebook_node(nb)
        out_html.write_text(body, encoding="utf-8")
    except Exception as e:
        # Conversion failed but execution succeeded
        return json.dumps({
            "ok": True,
            "executed_path": str(out_ipynb),
            "html_path": None,
            "warning": f"Conversione HTML fallita: {e}"
        }, ensure_ascii=False)

    return json.dumps({
        "ok": True,
        "executed_path": str(out_ipynb),
        "html_path": str(out_html)
    }, ensure_ascii=False)


@tool(name="nb_report", description="Genera un report riassuntivo HTML di un notebook SENZA eseguirlo.")
def nb_report(path: str, max_chars: int = 12000) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ip = Path(path)
    if not ip.exists() or ip.suffix.lower() != ".ipynb":
        return json.dumps({"ok": False, "error": "Notebook non trovato"}, ensure_ascii=False)

    try:
        import json as _json
        nb = _json.loads(ip.read_text(encoding="utf-8", errors="ignore"))
        cells = nb.get("cells", [])
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Errore lettura notebook: {e}"}, ensure_ascii=False)

    # Estrai contenuti testuali significativi
    md_parts: list[str] = []
    headings: list[str] = []
    code_outline: list[str] = []
    imports: list[str] = []

    for c in cells:
        ctype = c.get("cell_type")
        src = c.get("source", [])
        text = "".join(src) if isinstance(src, list) else (src or "")
        if ctype == "markdown":
            md_parts.append(text.strip())
            for line in text.splitlines():
                if line.strip().startswith(("#", "##", "###")):
                    headings.append(line.strip())
        elif ctype == "code":
            for line in text.splitlines():
                ls = line.strip()
                if ls.startswith("import ") or ls.startswith("from "):
                    imports.append(ls)
                if ls.startswith("def ") or ls.startswith("class "):
                    code_outline.append(ls)
                if ls.startswith("#") and len(code_outline) < 50:
                    code_outline.append(ls)

    # Estrai tabelle (output testuali con strutture tabulari semplici) e riferimenti a immagini
    try:
        outputs_tables: list[str] = []
        outputs_images: list[str] = []
        for c in cells:
            if c.get("cell_type") != "code":
                continue
            for out in c.get("outputs", []) or []:
                # testo tipo dataframe
                txt = "".join(out.get("text", [])) if isinstance(out.get("text"), list) else out.get("text")
                if txt and ("|" in txt or "\t" in txt):
                    outputs_tables.append(txt[:1000])
                # immagini
                data = out.get("data", {}) or {}
                if any(k.startswith("image/") for k in data.keys()):
                    outputs_images.append("Immagine generata (output cella)")
    except Exception:
        outputs_tables, outputs_images = [], []

    doc_text = "\n\n".join([
        "\n".join(headings[:80]),
        "\n\n".join(md_parts),
        "\n".join(imports[:120]),
        "\n".join(code_outline[:200]),
        "\n\nEstratti tabelle (grezzi):\n" + "\n\n".join(outputs_tables[:10]) if outputs_tables else "",
        "\n\nRiferimenti immagini: \n" + "\n".join(outputs_images[:20]) if outputs_images else "",
    ])
    doc_text = doc_text[: max_chars]

    # Prompt di sintesi strutturata in HTML
    prompt = (
        "Sei un assistente tecnico. Hai il contenuto di un notebook Jupyter. "
        "Produci un report HTML conciso e ben strutturato con le sezioni: "
        "Titolo, Obiettivi, Dataset/Prerequisiti, Panoramica passi, Punti chiave, "
        "Funzioni e componenti principali, Dipendenze/Import, Potenziali problemi, Prossimi passi. "
        "Non inventare; se un'informazione non c'è, scrivi 'Non specificato'. "
        "Usa liste puntate dove utile. Contenuto del notebook segue tra i tag <NOTEBOOK>...</NOTEBOOK>."
    )

    try:
        client = st.session_state.get("chat_client")
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAIClient(api_key=api_key, model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
        response = client.invoke(
            input=f"{prompt}\n<NOTEBOOK>\n{doc_text}\n</NOTEBOOK>",
            max_tokens=1500,
            temperature=0.2,
        )
        html_body = "".join([b.content for b in response.content if hasattr(b, "content")])
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Errore sintesi: {e}"}, ensure_ascii=False)

    from datetime import datetime
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", ip.stem)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_html = REPORTS_DIR / f"{safe_name}__{ts}.summary.html"
    try:
        out_html.write_text(html_body, encoding="utf-8")
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Errore salvataggio report: {e}"}, ensure_ascii=False)

    return json.dumps({"ok": True, "html_path": str(out_html)}, ensure_ascii=False)


def _parse_time_filters(q_lower: str) -> tuple[int | None, int | None]:
    ord_map = {
        "primo": 1,
        "prima": 1,
        "secondo": 2,
        "seconda": 2,
        "terzo": 3,
        "terza": 3,
        "quarto": 4,
        "quinta": 5,
        "sesto": 6,
        "settimo": 7,
        "ottavo": 8,
        "nono": 9,
        "decimo": 10,
    }
    week = None
    day = None
    m = re.search(r"week\s*(\d+)|settimana\s*(\d+)", q_lower)
    if m:
        week = int(m.group(1) or m.group(2))
    else:
        for k, v in ord_map.items():
            if re.search(rf"\b{k}\s+settimana\b", q_lower):
                week = v
                break
    m = re.search(r"day\s*(\d+)|giorno\s*(\d+)", q_lower)
    if m:
        day = int(m.group(1) or m.group(2))
    else:
        for k, v in ord_map.items():
            if re.search(rf"\b{k}\s+giorno\b", q_lower):
                day = v
                break
    return week, day


def _pre_filter_by_path(rows: list[dict], *, week: int | None, day: int | None) -> list[dict]:
    def ok(r):
        p = str(r.get("metadata", {}).get("source") or r.get("source", "")).lower()
        cond = True
        if week:
            cond = cond and (f"week {week}" in p)
        if day:
            cond = cond and (f"day {day}" in p or f"day {day} -" in p)
        return cond

    return [r for r in rows if ok(r)]


def _infer_ext_filter(q_lower: str) -> set[str] | None:
    # Heuristics: if the query mentions specific file types, restrict search
    if any(t in q_lower for t in ["notebook", "ipynb", ".ipynb"]):
        return {".ipynb"}
    if any(t in q_lower for t in ["slides", "slide", "presentazione", "deck", "pdf", ".pdf", ".pptx"]):
        return {".pdf", ".pptx"}
    if any(t in q_lower for t in ["python", ".py"]):
        return {".py"}
    return None


def _pre_filter_by_ext(rows: list[dict], allowed_exts: set[str] | None) -> list[dict]:
    if not allowed_exts:
        return rows
    out = []
    for r in rows:
        src = str(r.get("metadata", {}).get("source") or r.get("source", ""))
        for ext in allowed_exts:
            if src.lower().endswith(ext):
                out.append(r)
                break
    return out


def init_clients():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Set OPENAI_API_KEY in environment")
        st.stop()
    st.session_state["chat_client"] = OpenAIClient(api_key=api_key, model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    st.session_state["embed_client"] = OpenAIClient(api_key=api_key, model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))


def _save_preview_state():
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "open": st.session_state.get("preview_open", {}),
            "paths": st.session_state.get("preview_paths", {}),
        }
        PREVIEW_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def init_agent():
    system_prompt = (
        "Sei un assistant per knowledge base. Usa kb_search per recuperare contesto e cita le fonti. "
        "Quando l'utente chiede elenchi di file (notebook/slides/python), chiama kb_search con mode='files' e lascia che il filtro estensione sia inferito dalla query. "
        "Per i notebook: usa nb_list per individuarli e nb_report per generare un report HTML riassuntivo SENZA esecuzione. "
        "Restituisci sempre link file:// alle risorse per l'anteprima."
    )
    # restore memory from history
    history = st.session_state.get("history", [])
    mem = Memory()
    for role, content in history:
        block = TextBlock(content)
        mem.add_turn(blocks=block, role=ROLE.USER if role == "user" else ROLE.ASSISTANT)

    agent = Agent(
        name="hb-web-agent",
        system_prompt=system_prompt,
        client=st.session_state["chat_client"],
        tools=[kb_search, nb_list, nb_run, nb_report],
        terminate_on_text=True,
        memory=mem,
    )
    st.session_state["agent"] = agent


def main():
    st.set_page_config(page_title="Handbook Assistant", page_icon="📘", layout="wide")
    st.markdown(
        """
        <style>
        :root {
          --hb-bg1: #0b0f14;
          --hb-bg2: #0a0c0f;
          --hb-border: rgba(255,255,255,0.10);
          --hb-glass: rgba(255,255,255,0.06);
          --hb-text: #E8EBF0;
          --hb-subtext: #B6BCC6;
          --hb-accent: #88B1FF;
        }
        html, body, [data-testid="stAppViewContainer"] {
          background: radial-gradient(1400px 700px at 10% -10%, #0f1830 0%, transparent 70%),
                      radial-gradient(1000px 600px at 90% -20%, #102030 0%, transparent 65%),
                      linear-gradient(180deg, var(--hb-bg1) 0%, var(--hb-bg2) 100%) !important;
          color: var(--hb-text);
          font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Helvetica Neue", Helvetica, Arial, system-ui, "Segoe UI", Roboto, sans-serif;
        }
        [data-testid="stSidebar"] {
          background: rgba(255,255,255,0.02) !important;
          -webkit-backdrop-filter: saturate(180%) blur(18px);
          backdrop-filter: saturate(180%) blur(18px);
          border-right: 1px solid var(--hb-border);
        }
        .block-container { padding-top: 1.2rem; }
        .hb-hero {
          padding: 28px 32px; border-radius: 18px;
          background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
          -webkit-backdrop-filter: saturate(180%) blur(20px);
          backdrop-filter: saturate(180%) blur(20px);
          border: 1px solid var(--hb-border);
          box-shadow: 0 12px 40px rgba(0,0,0,0.35);
          margin: 8px 0 10px 0;
          position: relative;
          z-index: 1;
        }
        .hb-title { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.28; margin: 0; padding-top: 2px; }
        .hb-sub { color: var(--hb-subtext); margin-top: 4px; }
        .stButton button, .stDownloadButton button {
          border-radius: 12px; border: 1px solid var(--hb-border);
          background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
          color: var(--hb-text); font-weight: 600;
        }
        [data-testid="stChatMessage"] {
          background: rgba(255,255,255,0.04);
          border: 1px solid var(--hb-border);
          border-radius: 16px; padding: 12px 14px;
          box-shadow: 0 3px 12px rgba(0,0,0,0.25);
        }
        [data-testid="stMarkdownContainer"] p { font-size: 0.98rem; line-height: 1.58; }
        </style>
        <div class='hb-hero'>
          <div class='hb-title'>📘 Handbook Assistant</div>
          <div class='hb-sub'>RAG + Agents con DatapizzAI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chat_client" not in st.session_state:
        init_clients()
    if "index_rows" not in st.session_state:
        st.session_state["index_rows"] = load_index()
    if "history" not in st.session_state:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if HISTORY_FILE.exists():
                st.session_state["history"] = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            else:
                st.session_state["history"] = []
        except Exception:
            st.session_state["history"] = []
    if "last_sources" not in st.session_state:
        st.session_state["last_sources"] = []
    # rimosse variabili legacy della modalità Ricerca
    if "preview_open" not in st.session_state or "preview_paths" not in st.session_state:
        try:
            if PREVIEW_FILE.exists():
                data = json.loads(PREVIEW_FILE.read_text(encoding="utf-8"))
                st.session_state["preview_open"] = data.get("open", {})
                st.session_state["preview_paths"] = data.get("paths", {})
            else:
                st.session_state["preview_open"] = {}
                st.session_state["preview_paths"] = {}
        except Exception:
            st.session_state["preview_open"] = {}
            st.session_state["preview_paths"] = {}
    if "agent" not in st.session_state:
        init_agent()

    with st.sidebar:
        st.subheader("Indice")
        st.write(f"Chunk indicizzati: {len(st.session_state['index_rows'])}")
        if st.button("Ricarica indice"):
            st.session_state["index_rows"] = load_index()
            st.success("Indice ricaricato")
        if st.button("Pulisci indice"):
            try:
                if INDEX_FILE.exists():
                    INDEX_FILE.unlink()
                st.session_state["index_rows"] = []
                st.success("Indice pulito")
            except Exception as e:
                st.error(f"Impossibile pulire l'indice: {e}")
        # Sezione Agent spostata sotto la Drive ingestion (vedi più sotto)
        subdir = st.text_input("Sotto-cartella (opzionale)", placeholder="Week 1")
        if st.button("Esegui ingestion"):
            with st.spinner("Ingestion in corso..."):
                try:
                    # Run ingestion inline
                    from demo_minimo.handbook_assistant.ingest import main as ingest_main
                    ingest_main(subdir=subdir or None)
                    st.session_state["index_rows"] = load_index()
                    st.success("Ingestion completata e indice aggiornato")
                except Exception as e:
                    st.error(f"Errore ingestion: {e}")
        st.divider()
        st.subheader("Ingestion da Google Drive")
        st.caption("Richiede GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON e cartelle condivise con il service account")
        # Caricamento e configurazione rapida del Service Account JSON
        up = st.file_uploader("Carica service account JSON", type=["json"], accept_multiple_files=False)
        if up is not None:
            try:
                STATE_DIR.mkdir(parents=True, exist_ok=True)
                sa_path = STATE_DIR / "drive_sa.json"
                with open(sa_path, "wb") as f:
                    f.write(up.getbuffer())
                # prova a leggere info
                sa_obj = json.loads(sa_path.read_text(encoding="utf-8"))
                client_email = sa_obj.get("client_email", "?")
                os.environ["GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"] = str(sa_path)
                st.success(f"Service account configurato: {client_email}")
            except Exception as e:
                st.error(f"Errore nel salvataggio del JSON: {e}")

        drive_input = st.text_area("URL/ID cartelle (una per riga)", height=100, placeholder="https://drive.google.com/drive/folders/..\n1AbCdEf...")
        if st.button("Sincronizza da Drive e indicizza"):
            with st.spinner("Download da Drive e ingestion in corso..."):
                try:
                    from demo_minimo.handbook_assistant.ingest import sync_gdrive, main as ingest_main

                    urls = [u.strip() for u in (drive_input or "").splitlines() if u.strip()]
                    if not urls:
                        st.warning("Inserisci almeno una cartella Drive")
                    else:
                        num = sync_gdrive(urls)
                        st.info(f"Scaricati {num} file da Drive")
                        from demo_minimo.handbook_assistant.ingest import REMOTE_CACHE_DIR
                        ingest_main(subdir=None, root_override=REMOTE_CACHE_DIR)
                        st.session_state["index_rows"] = load_index()
                        st.success("Ingestion completata da Drive")
                except Exception as e:
                    st.error(f"Errore Drive: {e}")

        st.divider()
        st.subheader("Notebook Reporter (Agent)")
        nb_query = st.text_input("Cerca notebook", placeholder="es. Day 3 gemma")
        nb_root = st.selectbox("Sorgente", options=["auto", "data", "remote"], index=0)
        if st.button("Lista notebook"):
            try:
                lst_json = nb_list(nb_query, nb_root, 200)
                lst = json.loads(lst_json)
                st.session_state["nb_list"] = lst
                st.success(f"Trovati {len(lst)} notebook")
            except Exception as e:
                st.error(f"Errore lista notebook: {e}")
        if "_nb_expander_open" not in st.session_state:
            st.session_state["_nb_expander_open"] = {}
        if "_nb_last_report" not in st.session_state:
            st.session_state["_nb_last_report"] = {}
        for i, p in enumerate(st.session_state.get("nb_list", [])[:20]):
            exp_key = f"nbexp_{i}"
            with st.expander(f"Report → {Path(p).name}", expanded=bool(st.session_state["_nb_expander_open"].get(exp_key, False))):
                detail = st.radio("Dettaglio", options=["Più sintetico", "Più dettagliato"], horizontal=True, key=f"nb_detail_{i}")
                last_html = st.session_state["_nb_last_report"].get(exp_key)
                if last_html and Path(last_html).exists():
                    try:
                        file_bytes = Path(last_html).read_bytes()
                        st.download_button("⬇ Scarica report", data=file_bytes, file_name=Path(last_html).name, key=f"dl_report_persist_{i}")
                    except Exception as e:
                        st.caption(f"Download non disponibile: {e}")
                    if "_nb_share_links" not in st.session_state:
                        st.session_state["_nb_share_links"] = {}
                    with st.expander("Condividi pubblicamente"):
                        st.caption("Crea un link pubblico temporaneo (ngrok)")
                        ngrok_token = st.text_input("NGROK_AUTHTOKEN (opzionale)", type="password", key=f"ngrok_token_{i}")
                        if st.button("Crea link pubblico (ngrok)", key=f"share_ngrok_persist_{i}"):
                            try:
                                base_url, port = _ensure_report_server()
                                url = base_url + "/" + urllib.parse.quote(Path(last_html).name)
                                if ngrok_token:
                                    public = _ensure_ngrok(port, ngrok_token)
                                    url = public + "/" + urllib.parse.quote(Path(last_html).name)
                                st.success(f"Link: {url}")
                            except Exception as e:
                                st.error(f"Errore creazione link: {e}")
                        old = st.session_state["_nb_share_links"].get(exp_key)
                        if old:
                            st.caption(f"Ultimo link: {old}")
                if st.button("Produci report", key=f"report_nb_{i}"):
                    with st.spinner("Generazione report in corso..."):
                        try:
                            res = json.loads(nb_report(str(Path(p)), max_chars=24000 if detail=="Più dettagliato" else 12000))
                            if not res.get("ok"):
                                st.error(res.get("error", "Errore sconosciuto"))
                            else:
                                html_path = res.get("html_path")
                                msg = "Report generato. "
                                if html_path:
                                    file_url = "file://" + urllib.parse.quote(str(html_path), safe="/")
                                    msg += f"HTML: {file_url}"
                                st.session_state["history"].append(("assistant", msg))
                                if html_path:
                                    key = f"h{len(st.session_state['history'])-1}_0"
                                    st.session_state["preview_paths"][key] = html_path
                                    st.session_state["preview_open"][key] = True
                                    _save_preview_state()
                                    st.session_state["_nb_last_report"][exp_key] = html_path
                                    try:
                                        file_bytes = Path(html_path).read_bytes()
                                        st.download_button("⬇ Scarica report", data=file_bytes, file_name=Path(html_path).name, key=f"dl_report_{i}")
                                    except Exception as e:
                                        st.caption(f"Download non disponibile: {e}")
                                    with st.expander("Condividi pubblicamente"):
                                        st.caption("Crea un link pubblico temporaneo (ngrok)")
                                        ngrok_token2 = st.text_input("NGROK_AUTHTOKEN (opzionale)", type="password", key=f"ngrok_token2_{i}")
                                        if st.button("Crea link pubblico (ngrok)", key=f"share_ngrok_{i}"):
                                            try:
                                                base_url, port = _ensure_report_server()
                                                url = base_url + "/" + urllib.parse.quote(Path(html_path).name)
                                                if ngrok_token2:
                                                    public = _ensure_ngrok(port, ngrok_token2)
                                                    url = public + "/" + urllib.parse.quote(Path(html_path).name)
                                                st.success(f"Link: {url}")
                                            except Exception as e:
                                                st.error(f"Errore creazione link: {e}")
                                st.session_state["_nb_expander_open"][exp_key] = True
                                st.success("Completato")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Errore report: {e}")

        st.divider()
        st.subheader("Discord (beta)")
        st.caption("Legge canali pubblici (View/Read History). Richiede MESSAGE_CONTENT abilitato nelle impostazioni Bot.")
        # Precarica da secrets se disponibili
        try:
            _secrets_discord = st.secrets.get("discord", {}) if hasattr(st, "secrets") else {}
        except Exception:
            _secrets_discord = {}
        default_token = (_secrets_discord.get("bot_token") or "") if isinstance(_secrets_discord, dict) else ""
        default_channels = ",".join(_secrets_discord.get("channel_allowlist", [])) if isinstance(_secrets_discord, dict) else ""
        token = st.text_input("Bot token", type="password", value=default_token)
        ch_text = st.text_input("Channel IDs (separati da virgola)", value=default_channels, placeholder="111111111111111111,222222222222222222")
        max_msgs = st.number_input("Max messaggi per canale", min_value=50, max_value=5000, value=500, step=50)
        if st.button("Backfill Discord e indicizza"):
            ch_ids = [c.strip() for c in (ch_text or "").split(",") if c.strip()]
            if not token or not ch_ids:
                st.warning("Inserisci token e almeno un Channel ID")
            else:
                with st.spinner("Backfill Discord in corso..."):
                    try:
                        added = ingest_discord(token, ch_ids, max_messages_per_channel=int(max_msgs))
                        st.info(f"Indicizzati {added} chunk da Discord")
                        st.session_state["index_rows"] = load_index()
                        st.success("Indice aggiornato")
                    except Exception as e:
                        st.error(f"Errore Discord: {e}")

    st.divider()
    # Chat unica modalità
    if "history" not in st.session_state:
        st.session_state["history"] = []
    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Pulisci chat"):
            st.session_state["history"] = []
            try:
                HISTORY_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                PREVIEW_FILE.unlink(missing_ok=True)
            except Exception:
                pass
            st.session_state["preview_open"] = {}
            st.session_state["preview_paths"] = {}
            init_agent()
    history = st.session_state["history"]
    total_msgs = len(history)
    for disp_idx, (role, content) in enumerate(reversed(history)):
        real_idx = total_msgs - 1 - disp_idx
        msg_col, prev_col = st.columns([3, 2])
        with msg_col:
            st.chat_message(role).write(content)
            sources = _extract_sources(content) if role == "assistant" else []
            if sources:
                st.caption("Fonti citate")
            for j, p in enumerate(sources):
                key = f"h{real_idx}_{j}"
                st.session_state["preview_paths"][key] = p
                is_open = bool(st.session_state["preview_open"].get(key, False))
                nm, ic = st.columns([20, 1])
                with nm:
                    name = _truncate_middle(Path(p).name, 64)
                    st.markdown("<div style='white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>" + name + "</div>", unsafe_allow_html=True)
                with ic:
                    icon = "🔎" if not is_open else "✖"
                    if st.button(icon, key=f"ico_{key}", help="Apri/Chiudi preview", use_container_width=True):
                        st.session_state["preview_open"][key] = not is_open
                        _save_preview_state()
                        st.rerun()
        with prev_col:
            # Mostra le anteprime solo per questo messaggio, allineate lateralmente
            sources = _extract_sources(content) if role == "assistant" else []
            for j in range(len(sources)):
                key = f"h{real_idx}_{j}"
                if not st.session_state["preview_open"].get(key, False):
                    continue
                path = st.session_state["preview_paths"].get(key)
                if not path:
                    continue
                _render_preview(path)
                try:
                    file_bytes = Path(path).read_bytes()
                    st.download_button("⬇ Scarica", data=file_bytes, file_name=Path(path).name, key=f"dl_{key}")
                except Exception as e:
                    st.caption(f"Download non disponibile: {e}")
                if st.button("Chiudi", key=f"close_{key}"):
                    st.session_state["preview_open"][key] = False
                    _save_preview_state()
                    st.rerun()

    user = st.chat_input("Fai una domanda...")
    if user:
        st.session_state["history"].append(("user", user))
        st.chat_message("user").write(user)
        # also update memory persistently
        try:
            st.session_state["agent"]._memory.add_turn(TextBlock(user), ROLE.USER)
        except Exception:
            pass
        res = st.session_state["agent"].run(user, tool_choice="required_first")
        st.session_state["history"].append(("assistant", res))
        st.chat_message("assistant").write(res)
        # update preview state for new answer
        sources = _extract_sources(res)
        st.session_state["last_sources"] = sources
        if sources:
            for j, p in enumerate(sources):
                key = f"h{len(st.session_state['history'])-1}_{j}"
                st.session_state["preview_paths"][key] = p
        try:
            st.session_state["agent"]._memory.add_turn(TextBlock(res), ROLE.ASSISTANT)
        except Exception:
            pass
        # persist
        try:
            HISTORY_FILE.write_text(json.dumps(st.session_state["history"], ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        _save_preview_state()

    # No extra expander; handled per-messaggio e persistente con preview_open


def _extract_sources(text: str) -> list[str]:
    # Preserva ordine e rimuove duplicati mantenendo la prima occorrenza
    matches: list[tuple[int, str]] = []
    for m in re.finditer(r"file://([^\s\)]+)", text):
        p = urllib.parse.unquote(m.group(1))
        p = "/" + p.lstrip("/") if not p.startswith("/") else p
        matches.append((m.start(), p))
    for m in re.finditer(r"`(/[^`]+)`", text):
        matches.append((m.start(), m.group(1)))
    matches.sort(key=lambda x: x[0])
    ordered: list[str] = []
    seen: set[str] = set()
    for _, p in matches:
        if p in seen:
            continue
        if Path(p).exists():
            ordered.append(p)
            seen.add(p)
    return ordered


def _truncate_middle(text: str, max_len: int = 64) -> str:
    if len(text) <= max_len:
        return text
    keep = max_len - 1
    head = max(keep // 2, 1)
    tail = max(keep - head, 1)
    return text[:head] + "…" + text[-tail:]


def _render_preview(path: str):
    p = Path(path)
    st.markdown(f"### Preview: {p.name}")
    if not p.exists():
        st.warning("File non trovato")
        return
    ext = p.suffix.lower()
    try:
        if ext == ".html":
            try:
                html = p.read_text(encoding="utf-8", errors="ignore")
                components.html(html, height=600, scrolling=True)
                return
            except Exception:
                pass
        if ext == ".ipynb":
            preview = _get_ipynb_preview(str(p), max_cells=8, mtime=os.path.getmtime(p))
            for kind, content in preview:
                if kind == "md":
                    st.markdown(content)
                elif kind == "code":
                    st.code(content, language="python")
        elif ext == ".py":
            st.code(p.read_text(encoding="utf-8", errors="ignore"), language="python")
        elif ext in {".md", ".txt"}:
            st.text(p.read_text(encoding="utf-8", errors="ignore")[:4000])
        elif ext == ".pdf":
            # Render 1 pagina cache-ata, per velocizzare
            imgs = _get_pdf_preview(str(p), max_pages=1, zoom=2.0, mtime=os.path.getmtime(p))
            if imgs:
                for i, img in enumerate(imgs, start=1):
                    st.image(img, caption=f"Pagina {i}")
                if len(imgs) == 1:
                    st.caption("Anteprima limitata alla prima pagina")
            else:
                st.info("Anteprima PDF non disponibile. Usa il pulsante Scarica per aprirlo localmente.")
        elif ext in {".pptx", ".docx", ".html", ".htm"}:
            from demo_minimo.handbook_assistant.ingest import load_text
            st.text(load_text(p)[:4000])
        else:
            st.text("Preview non supportata; prova ad aprire il file localmente.")
    except Exception as e:
        st.error(f"Impossibile generare la preview: {e}")


@st.cache_data(show_spinner=False)
def _get_ipynb_preview(path: str, max_cells: int, mtime: float):
    import json as _json
    nb = _json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
    cells = nb.get("cells", [])
    out = []
    shown = 0
    for c in cells:
        if c.get("cell_type") == "markdown":
            out.append(("md", "".join(c.get("source", []))))
        elif c.get("cell_type") == "code":
            out.append(("code", "".join(c.get("source", []))))
        shown += 1
        if shown >= max_cells:
            break
    return out


@st.cache_data(show_spinner=False)
def _get_pdf_preview(path: str, max_pages: int, zoom: float, mtime: float):
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    imgs = []
    with fitz.open(path) as doc:
        pages = min(max_pages, len(doc))
        for i in range(pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            imgs.append(pix.tobytes("png"))
    return imgs


def _upload_transfer_sh(path: str) -> str:
    p = Path(path)
    url = "https://transfer.sh/" + urllib.parse.quote(p.name)
    data = p.read_bytes()
    req = urllib.request.Request(url, data=data, method="PUT")
    req.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(req) as resp:
        link = resp.read().decode("utf-8").strip()
    return link


def _upload_0x0(path: str) -> str:
    try:
        import requests
    except Exception:
        raise RuntimeError("Installa 'requests' per usare 0x0.st oppure usa transfer.sh")
    p = Path(path)
    files = {"file": (p.name, p.read_bytes(), "text/html")}
    r = requests.post("https://0x0.st", files=files, timeout=30)
    r.raise_for_status()
    return r.text.strip()


def _get_free_port(preferred: int = 8008) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def _ensure_report_server() -> tuple[str, int]:
    # Returns (base_url, port)
    if "_report_server" in st.session_state and st.session_state["_report_server"]:
        return st.session_state.get("_report_base", "http://127.0.0.1:" + str(st.session_state.get("_report_port", 0))), st.session_state.get("_report_port", 0)
    port = _get_free_port(8008)
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    handler = partial(SimpleHTTPRequestHandler, directory=str(REPORTS_DIR))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)

    def _run():
        try:
            httpd.serve_forever()
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    st.session_state["_report_server"] = t
    st.session_state["_report_port"] = port
    st.session_state["_report_base"] = f"http://127.0.0.1:{port}"
    return st.session_state["_report_base"], port


def _ensure_ngrok(port: int, token: str) -> str:
    try:
        from pyngrok import ngrok
    except Exception:
        raise RuntimeError("Installa pyngrok: pip install pyngrok")
    if token:
        try:
            ngrok.set_auth_token(token)
        except Exception:
            pass
    # Reuse existing tunnel if present
    if st.session_state.get("_ngrok_url"):
        return st.session_state["_ngrok_url"]
    public_url = ngrok.connect(addr=port, proto="http").public_url
    st.session_state["_ngrok_url"] = public_url
    return public_url


if __name__ == "__main__":
    main()


