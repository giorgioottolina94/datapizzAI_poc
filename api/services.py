import json
import os
import re
import threading
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datapizzai.agents import Agent
from datapizzai.clients.openai_client import OpenAIClient
from datapizzai.memory import Memory
from datapizzai.tools.tools import tool, Tool
from datapizzai.type.type import ROLE, TextBlock


BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_FILE = BASE_DIR / "index" / "chunks.jsonl"
STATE_DIR = BASE_DIR / "state"
HISTORY_FILE = STATE_DIR / "history.json"
INGESTION_HISTORY_FILE = STATE_DIR / "ingestion_history.json"
NOTEBOOK_HISTORY_FILE = STATE_DIR / "notebook_history.json"
PREVIEW_FILE = STATE_DIR / "preview_state.json"
REPORTS_DIR = BASE_DIR / "reports"
OPENAI_KEY_FILE = STATE_DIR / "openai_key.txt"
DRIVE_SA_FILE = STATE_DIR / "drive_sa.json"


_lock = threading.Lock()
_clients: Dict[str, Any] = {}
_ingest_state: Dict[str, Any] = {
    "job_id": None,
    "status": "idle",  # idle|downloading|indexing|done|error
    "downloaded": 0,
    "chunks_before": None,
    "chunks_after": None,
    "error": None,
    "last_update": None,
}


def get_chat_client() -> OpenAIClient:
    with _lock:
        cli = _clients.get("chat")
        if cli is None:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key and OPENAI_KEY_FILE.exists():
                try:
                    api_key = OPENAI_KEY_FILE.read_text(encoding="utf-8").strip()
                    if api_key:
                        os.environ["OPENAI_API_KEY"] = api_key
                except Exception:
                    api_key = ""
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY non configurata")
            model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            cli = OpenAIClient(api_key=api_key, model=model)
            _clients["chat"] = cli
        return cli


def get_embed_client() -> OpenAIClient:
    with _lock:
        cli = _clients.get("embed")
        if cli is None:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key and OPENAI_KEY_FILE.exists():
                try:
                    api_key = OPENAI_KEY_FILE.read_text(encoding="utf-8").strip()
                    if api_key:
                        os.environ["OPENAI_API_KEY"] = api_key
                except Exception:
                    api_key = ""
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY non configurata")
            model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            cli = OpenAIClient(api_key=api_key, model=model)
            _clients["embed"] = cli
        return cli


def set_openai_api_key(api_key: str) -> None:
    """Set OPENAI_API_KEY at runtime and reset cached clients."""
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("API key non valida")
    with _lock:
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            OPENAI_KEY_FILE.write_text(api_key.strip(), encoding="utf-8")
        except Exception:
            pass
        # Reset cached clients so next call uses the updated key
        _clients.pop("chat", None)
        _clients.pop("embed", None)


def config_state() -> Dict[str, Any]:
    """Return minimal config health used by the UI."""
    sa_path = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
    sa_present = bool(sa_path and Path(sa_path).exists()) or DRIVE_SA_FILE.exists()
    return {
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY")) or OPENAI_KEY_FILE.exists(),
        "drive_sa_present": sa_present,
    }


def ensure_drive_sa_env() -> None:
    """If the Drive SA env is missing but default file exists, set it."""
    if not os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON") and DRIVE_SA_FILE.exists():
        os.environ["GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"] = str(DRIVE_SA_FILE)


def ensure_openai_key_env() -> None:
    """If the OpenAI API key env is missing but default file exists, set it."""
    if not os.getenv("OPENAI_API_KEY") and OPENAI_KEY_FILE.exists():
        try:
            api_key = OPENAI_KEY_FILE.read_text(encoding="utf-8").strip()
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        except Exception:
            pass


def start_drive_ingest(urls: List[str]) -> str:
    """Start Drive sync+ingest in background and return a job id."""
    job_id = f"drive-{uuid.uuid4().hex[:8]}"
    
    # Add initial record to history
    _add_ingestion_record("drive", {
        "job_id": job_id,
        "urls": urls,
        "url_count": len(urls)
    }, "started")

    def _run():
        from demo_minimo.handbook_assistant.ingest import sync_gdrive, main as ingest_main, REMOTE_CACHE_DIR
        try:
            ensure_drive_sa_env()
            chunks_before = len(load_index())
            with _lock:
                _ingest_state.update({
                    "job_id": job_id,
                    "status": "downloading",
                    "downloaded": 0,
                    "chunks_before": chunks_before,
                    "chunks_after": None,
                    "error": None,
                    "last_update": datetime.utcnow().isoformat(),
                })
            
            # Update history status
            _update_ingestion_record(job_id, {
                "status": "downloading",
                "chunks_before": chunks_before
            })
            
            num = sync_gdrive(urls)
            with _lock:
                _ingest_state.update({
                    "downloaded": int(num),
                    "status": "indexing",
                    "last_update": datetime.utcnow().isoformat(),
                })
            
            # Update history with download results
            _update_ingestion_record(job_id, {
                "status": "indexing",
                "downloaded": int(num)
            })
            
            ensure_openai_key_env()  # Load OpenAI key before indexing
            ingest_main(subdir=None, root_override=REMOTE_CACHE_DIR)
            chunks_after = len(load_index())
            
            with _lock:
                _ingest_state.update({
                    "chunks_after": chunks_after,
                    "status": "done",
                    "last_update": datetime.utcnow().isoformat(),
                })
            
            # Update history with final results
            _update_ingestion_record(job_id, {
                "status": "completed",
                "chunks_after": chunks_after,
                "added_chunks": chunks_after - chunks_before
            })
            
        except Exception as e:
            with _lock:
                _ingest_state.update({
                    "status": "error",
                    "error": str(e),
                    "last_update": datetime.utcnow().isoformat(),
                })
            
            # Update history with error
            _update_ingestion_record(job_id, {
                "status": "error",
                "error": str(e)
            })

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    with _lock:
        _ingest_state["thread"] = t
    return job_id


def start_local_ingest(subdir: Optional[str] = None) -> str:
    """Start local ingest in background and return a job id."""
    job_id = f"local-{uuid.uuid4().hex[:8]}"
    
    # Add initial record to history
    _add_ingestion_record("local", {
        "job_id": job_id,
        "subdir": subdir or "all"
    }, "started")

    def _run():
        from demo_minimo.handbook_assistant.ingest import main as ingest_main
        try:
            chunks_before = len(load_index())
            with _lock:
                _ingest_state.update({
                    "job_id": job_id,
                    "status": "indexing",
                    "downloaded": 0,
                    "chunks_before": chunks_before,
                    "chunks_after": None,
                    "error": None,
                    "last_update": datetime.utcnow().isoformat(),
                })
            
            # Update history status
            _update_ingestion_record(job_id, {
                "status": "indexing",
                "chunks_before": chunks_before
            })
            
            ensure_openai_key_env()  # Load OpenAI key before indexing
            ingest_main(subdir=subdir)
            chunks_after = len(load_index())
            
            with _lock:
                _ingest_state.update({
                    "chunks_after": chunks_after,
                    "status": "done",
                    "last_update": datetime.utcnow().isoformat(),
                })
            
            # Update history with final results
            _update_ingestion_record(job_id, {
                "status": "completed",
                "chunks_after": chunks_after,
                "added_chunks": chunks_after - chunks_before
            })
            
        except Exception as e:
            with _lock:
                _ingest_state.update({
                    "status": "error",
                    "error": str(e),
                    "last_update": datetime.utcnow().isoformat(),
                })
            
            # Update history with error
            _update_ingestion_record(job_id, {
                "status": "error",
                "error": str(e)
            })

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    with _lock:
        _ingest_state["thread"] = t
    return job_id


def get_ingest_state(job_id: Optional[str] = None) -> Dict[str, Any]:
    with _lock:
        st = dict(_ingest_state)
    if job_id and st.get("job_id") != job_id:
        # No such job; return idle
        return {"job_id": job_id, "status": "idle"}
    # Don't expose thread
    st.pop("thread", None)
    return st


def load_index() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def cosine(a: List[float], b: List[float]) -> float:
    import math

    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def embed_query(query: str) -> List[float]:
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    client = get_embed_client()
    return client.embed(query, model)


def _parse_time_filters(q_lower: str) -> Tuple[Optional[int], Optional[int]]:
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


def _infer_ext_filter(q_lower: str) -> Optional[set[str]]:
    if any(t in q_lower for t in ["notebook", "ipynb", ".ipynb"]):
        return {".ipynb"}
    if any(t in q_lower for t in ["slides", "slide", "presentazione", "deck", "pdf", ".pdf", ".pptx"]):
        return {".pdf", ".pptx"}
    if any(t in q_lower for t in ["python", ".py"]):
        return {".py"}
    return None


def _pre_filter_by_path(rows: List[Dict[str, Any]], *, week: Optional[int], day: Optional[int]) -> List[Dict[str, Any]]:
    def ok(r: Dict[str, Any]) -> bool:
        p = str(r.get("metadata", {}).get("source") or r.get("source", "")).lower()
        cond = True
        if week:
            cond = cond and (f"week {week}" in p)
        if day:
            cond = cond and (f"day {day}" in p or f"day {day} -" in p)
        return cond

    return [r for r in rows if ok(r)]


def _pre_filter_by_ext(rows: List[Dict[str, Any]], allowed_exts: Optional[set[str]]) -> List[Dict[str, Any]]:
    if not allowed_exts:
        return rows
    out: List[Dict[str, Any]] = []
    for r in rows:
        src = str(r.get("metadata", {}).get("source") or r.get("source", ""))
        for ext in allowed_exts:
            if src.lower().endswith(ext):
                out.append(r)
                break
    return out


@tool(name="kb_search", description="Cerca nei documenti indicizzati e restituisce passaggi con citazioni")
def kb_search(query: str, k: int = 5, mode: Optional[str] = None) -> str:
    rows = load_index()
    q_lower = query.lower()
    if mode == "files" or any(t in q_lower for t in ["file", "notebook", "ipynb", "esercizi", "exercise", ".ipynb", "slides", "slide", "presentazione", "pdf", ".pdf", ".pptx", "python", ".py"]):
        week, day = _parse_time_filters(q_lower)
        allowed_exts = _infer_ext_filter(q_lower)
        candidates = rows
        if week or day:
            candidates = _pre_filter_by_path(candidates, week=week, day=day)
        candidates = _pre_filter_by_ext(candidates, allowed_exts)

        def score_name(r: Dict[str, Any]) -> int:
            name = str(r.get("metadata", {}).get("source") or r.get("source", "")).lower()
            score = 0
            for tok in ["exercise", "eserc", "slides", "slide", "presentazione"]:
                if tok in name:
                    score += 1
            if q_lower in name:
                score += 1
            return score

        seen = set()
        dedup: List[Dict[str, Any]] = []
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
        out: List[Dict[str, Any]] = []
        for r in ranked[: max(k, 10)]:
            src = r.get("metadata", {}).get("source") or r.get("source")
            kind = Path(src).suffix.lower().lstrip(".")
            out.append({
                "source": src,
                "kind": kind,
            })
        return json.dumps(out, ensure_ascii=False)

    # Dense search
    q = embed_query(query)
    scored = [(cosine(q, r.get("embedding", [])), r) for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    seen_src: set[str] = set()
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
    return payload[:6000]


@tool(name="nb_list", description="Elenca notebook .ipynb nelle cartelle locali. Restituisce JSON con percorsi.")
def nb_list(query: str = "", root: str = "auto", limit: int = 200) -> str:
    from demo_minimo.handbook_assistant.ingest import DATA_DIR, REMOTE_CACHE_DIR
    roots: List[Path]
    if root == "data":
        roots = [DATA_DIR]
    elif root == "remote":
        roots = [REMOTE_CACHE_DIR]
    else:
        roots = [REMOTE_CACHE_DIR, DATA_DIR]

    q = (query or "").lower().strip()
    week, day = _parse_time_filters(q)
    results: List[str] = []
    for base in roots:
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
            for fn in filenames:
                if not fn.lower().endswith(".ipynb"):
                    continue
                p = str(Path(dirpath) / fn)
                pl = p.lower()
                # If week/day inferred, require them in path as english tokens
                if week is not None and f"week {week}" not in pl:
                    continue
                if day is not None and (f"day {day}" not in pl and f"day {day} -" not in pl):
                    continue
                # Fallback substring filter if free-text provided and no week/day extracted
                if (week is None and day is None) and q and q not in pl:
                    continue
                results.append(p)
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break
    return json.dumps(results, ensure_ascii=False)


@tool(name="nb_run", description="Esegue un notebook e genera un report HTML. Accetta params_json opzionale.")
def nb_run(path: str, params_json: Optional[str] = None, timeout_s: int = 900) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ip = Path(path)
    if not ip.exists() or ip.suffix.lower() != ".ipynb":
        return json.dumps({"ok": False, "error": "Notebook non trovato"}, ensure_ascii=False)
    try:
        params = json.loads(params_json) if params_json else {}
        if not isinstance(params, dict):
            raise ValueError("params_json deve essere un oggetto JSON")
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Parametri non validi: {e}"}, ensure_ascii=False)

    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", ip.stem)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_ipynb = REPORTS_DIR / f"{safe_name}__{ts}.executed.ipynb"
    out_html = REPORTS_DIR / f"{safe_name}__{ts}.report.html"

    try:
        import papermill as pm
        import nbformat
        
        # Pre-process notebook to handle Colab-specific imports
        try:
            nb = nbformat.read(str(ip), as_version=4)
            
            # Add a cell at the beginning to handle missing Colab modules
            colab_fix_cell = nbformat.v4.new_code_cell(source="""
# Fix for Google Colab imports in local environment
import sys
from unittest.mock import MagicMock

# Mock google.colab module if not available
if 'google.colab' not in sys.modules:
    sys.modules['google.colab'] = MagicMock()
    sys.modules['google.colab.drive'] = MagicMock()
    sys.modules['google.colab.files'] = MagicMock()
    
# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
""")
            nb.cells.insert(0, colab_fix_cell)
            
            # Save the preprocessed notebook
            temp_nb_path = str(ip.parent / f"temp_{ip.name}")
            nbformat.write(nb, temp_nb_path)
            
        except Exception:
            # If preprocessing fails, use original notebook
            temp_nb_path = str(ip)
        
        _orig_cwd = os.getcwd()
        try:
            pm.execute_notebook(
                temp_nb_path,
                str(out_ipynb),
                parameters=params,
                cwd=str(ip.parent),
                kernel_name="python3",
                progress_bar=False,
                request_save_on_cell_execute=True,
                execution_timeout=timeout_s,
            )
        finally:
            try:
                os.chdir(_orig_cwd)
                # Clean up temp file if it was created
                if temp_nb_path != str(ip) and Path(temp_nb_path).exists():
                    Path(temp_nb_path).unlink()
            except Exception:
                pass
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Errore esecuzione: {e}"}, ensure_ascii=False)

    try:
        import nbformat
        from nbconvert import HTMLExporter
        nb = nbformat.read(str(out_ipynb), as_version=4)
        exporter = HTMLExporter()
        body, _ = exporter.from_notebook_node(nb)
        out_html.write_text(body, encoding="utf-8")
    except Exception as e:
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
def ai_report_for_file(path: str, max_chars: int = 12000) -> str:
    """Generate an AI-written modern HTML report for a generic file.
    
    - .ipynb: delega a nb_report
    - .md/.txt/.py/.csv/.json: taglia a max_chars
    - .pdf: estrai testo con PyMuPDF (se disponibile)
    - .pptx/.docx: prova ingest loader se presente
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ip = Path(path)
    if not ip.exists():
        return json.dumps({"ok": False, "error": "File non trovato"}, ensure_ascii=False)

    ext = ip.suffix.lower()
    if ext == ".ipynb":
        # Per i notebook usa la catena di report dedicata
        return nb_report(path, max_chars)

    # For other files, extract text and generate a generic report
    content = ""
    notebook_name = ip.name
    if ext in {".md", ".txt", ".py", ".csv", ".json"}:
        content = ip.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    elif ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            with fitz.open(ip) as doc:  # type: ignore[attr-defined]
                for page in doc:
                    content += page.get_text()
                    if len(content) >= max_chars:
                        break
            content = content[:max_chars]
        except Exception:
            content = "Impossibile estrarre testo dal PDF."
    elif ext in {".pptx", ".docx"}:
        try:
            from demo_minimo.handbook_assistant.ingest import load_text
            content = load_text(ip)[:max_chars]
        except Exception:
            content = "Impossibile estrarre testo dal file Office."
    else:
        content = "Tipo di file non supportato per il report AI."

    # Generate a modern HTML report from the extracted content
    html_content = generate_modern_report_html(content, notebook_name, datetime.now().strftime('%d/%m/%Y alle %H:%M'))
    report_filename = f"{ip.stem}__{datetime.now().strftime('%Y%m%d-%H%M%S')}.summary.html"
    report_path = REPORTS_DIR / report_filename
    report_path.write_text(html_content, encoding="utf-8")

    return json.dumps({"ok": True, "html_path": str(report_path), "url": f"/reports/{report_filename}"}, ensure_ascii=False)


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

    md_parts: List[str] = []
    headings: List[str] = []
    code_outline: List[str] = []
    imports: List[str] = []

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

    try:
        outputs_tables: List[str] = []
        outputs_images: List[str] = []
        for c in cells:
            if c.get("cell_type") != "code":
                continue
            for out in c.get("outputs", []) or []:
                txt = "".join(out.get("text", [])) if isinstance(out.get("text"), list) else out.get("text")
                if txt and ("|" in txt or "\t" in txt):
                    outputs_tables.append(txt[:1000])
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

    prompt = (
        "Sei un assistente tecnico. Hai il contenuto di un notebook Jupyter. "
        "Produci un report HTML conciso e ben strutturato con le sezioni: "
        "Titolo, Obiettivi, Dataset/Prerequisiti, Panoramica passi, Punti chiave, "
        "Funzioni e componenti principali, Dipendenze/Import, Potenziali problemi, Prossimi passi. "
        "Non inventare; se un'informazione non c'è, scrivi 'Non specificato'. "
        "Usa liste puntate dove utile. Contenuto del notebook segue tra i tag <NOTEBOOK>...</NOTEBOOK>."
    )

    try:
        client = get_chat_client()
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
    
    # Create modern HTML template with embedded CSS
    full_html = generate_modern_report_html(html_body, ip.name, ts)
    
    try:
        out_html.write_text(full_html, encoding="utf-8")
    except Exception as e:
        return json.dumps({"ok": False, "error": f"Errore salvataggio report: {e}"}, ensure_ascii=False)

    return json.dumps({"ok": True, "html_path": str(out_html)}, ensure_ascii=False)


def generate_modern_report_html(content: str, notebook_name: str, timestamp: str) -> str:
    """Generate a modern HTML report with embedded CSS and responsive design."""
    
    # Extract title from content if present
    title = f"Report: {notebook_name}"
    if "# " in content:
        first_heading = content.split("# ")[1].split("\n")[0].strip()
        if first_heading:
            title = f"📓 {first_heading}"
    
    return f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        :root {{
            --bg-primary: #0a0b0f;
            --bg-secondary: #131318;
            --bg-tertiary: #1d1e24;
            --text-primary: #ffffff;
            --text-secondary: #b8bcc8;
            --text-muted: #6b7280;
            --accent: #8b5cf6;
            --accent-hover: #a855f7;
            --border: rgba(139, 92, 246, 0.1);
            --border-bright: rgba(139, 92, 246, 0.3);
            --success: #10b981;
            --warning: #f59e0b;
            --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.15);
            --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.2);
            --shadow-lg: 0 12px 40px rgba(0, 0, 0, 0.3);
        }}
        
        * {{ box-sizing: border-box; }}
        
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            animation: fadeIn 0.8s ease-out;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .header {{
            background: linear-gradient(135deg, 
                rgba(139, 92, 246, 0.1), 
                rgba(168, 85, 247, 0.05)
            );
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-bright);
            border-radius: 20px;
            padding: 30px 40px;
            margin-bottom: 30px;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, 
                transparent, 
                var(--accent), 
                var(--accent-hover),
                transparent
            );
        }}
        
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.2em;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent), var(--accent-hover));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header .meta {{
            color: var(--text-muted);
            font-size: 0.9em;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 15px;
        }}
        
        .badge {{
            background: rgba(139, 92, 246, 0.2);
            color: var(--accent);
            padding: 4px 12px;
            border-radius: 8px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .content {{
            background: linear-gradient(135deg, 
                rgba(255,255,255,0.08), 
                rgba(255,255,255,0.02)
            );
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 40px;
            box-shadow: var(--shadow-md);
            position: relative;
        }}
        
        .content h1, .content h2, .content h3, .content h4, .content h5, .content h6 {{
            color: var(--text-primary);
            margin-top: 2em;
            margin-bottom: 1em;
            font-weight: 600;
            position: relative;
        }}
        
        .content h2 {{
            font-size: 1.5em;
            border-left: 4px solid var(--accent);
            padding-left: 20px;
            margin-left: -24px;
        }}
        
        .content h3 {{
            font-size: 1.3em;
            color: var(--accent);
        }}
        
        .content p {{
            color: var(--text-secondary);
            margin: 1em 0;
        }}
        
        .content ul, .content ol {{
            color: var(--text-secondary);
            padding-left: 1.5em;
        }}
        
        .content li {{
            margin: 0.5em 0;
            position: relative;
        }}
        
        .content ul li::marker {{
            color: var(--accent);
        }}
        
        .content code {{
            background: rgba(139, 92, 246, 0.1);
            color: var(--accent-hover);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
        }}
        
        .content pre {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            overflow-x: auto;
            margin: 1.5em 0;
            box-shadow: var(--shadow-sm);
        }}
        
        .content pre code {{
            background: none;
            color: var(--text-secondary);
            padding: 0;
        }}
        
        .content blockquote {{
            border-left: 4px solid var(--warning);
            background: rgba(245, 158, 11, 0.1);
            padding: 15px 20px;
            margin: 1.5em 0;
            border-radius: 0 8px 8px 0;
            color: var(--text-secondary);
        }}
        
        .content table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5em 0;
            background: rgba(255,255,255,0.02);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }}
        
        .content th, .content td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        .content th {{
            background: rgba(139, 92, 246, 0.1);
            color: var(--text-primary);
            font-weight: 600;
        }}
        
        .content td {{
            color: var(--text-secondary);
        }}
        
        .content strong {{
            color: var(--text-primary);
            font-weight: 600;
        }}
        
        .content em {{
            color: var(--accent-hover);
            font-style: italic;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-muted);
            font-size: 0.9em;
            border-top: 1px solid var(--border);
            margin-top: 40px;
        }}
        
        .print-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, var(--accent), var(--accent-hover));
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: var(--shadow-md);
            transition: all 0.3s ease;
            font-size: 0.9em;
        }}
        
        .print-button:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 15px; }}
            .header {{ padding: 20px; }}
            .content {{ padding: 25px; }}
            .header h1 {{ font-size: 1.8em; }}
            .header .meta {{ flex-direction: column; align-items: flex-start; gap: 10px; }}
            .print-button {{ position: static; margin: 20px auto; display: block; }}
        }}
        
        @media print {{
            body {{ background: white; color: black; }}
            .header, .content {{ 
                background: white; 
                border: 1px solid #ddd; 
                box-shadow: none; 
            }}
            .print-button {{ display: none; }}
        }}
    </style>
</head>
<body>
    <button class="print-button" onclick="window.print()">🖨️ Stampa</button>
    
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="meta">
                <div>
                    <strong>Notebook:</strong> {notebook_name}<br>
                    <strong>Generato:</strong> {datetime.now().strftime('%d/%m/%Y alle %H:%M')}
                </div>
                <div class="badge">Jupyter Report</div>
            </div>
        </div>
        
        <div class="content">
            {content}
        </div>
        
        <div class="footer">
            <p>📊 Report generato automaticamente dal <strong>DatapizzAI Notebook Explorer</strong></p>
            <p>🤖 Assistente AI per l'analisi e documentazione di Jupyter Notebooks</p>
        </div>
    </div>
    
    <script>
        // Add smooth scrolling for internal links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{ behavior: 'smooth' }});
                }}
            }});
        }});
        
        // Add copy button to code blocks
        document.querySelectorAll('pre code').forEach((block) => {{
            const button = document.createElement('button');
            button.textContent = '📋 Copia';
            button.style.cssText = `
                position: absolute;
                top: 10px;
                right: 10px;
                background: var(--accent);
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.8em;
                opacity: 0;
                transition: opacity 0.3s ease;
            `;
            
            const pre = block.parentElement;
            pre.style.position = 'relative';
            pre.appendChild(button);
            
            pre.addEventListener('mouseenter', () => button.style.opacity = '1');
            pre.addEventListener('mouseleave', () => button.style.opacity = '0');
            
            button.addEventListener('click', async () => {{
                try {{
                    await navigator.clipboard.writeText(block.textContent);
                    button.textContent = '✅ Copiato!';
                    setTimeout(() => button.textContent = '📋 Copia', 2000);
                }} catch (err) {{
                    console.error('Failed to copy: ', err);
                }}
            }});
        }});
    </script>
</body>
</html>"""




def _load_history() -> List[Tuple[str, str, List[str]]]:
    """Load history with sources. Returns (role, text, sources) tuples."""
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        result = []
        needs_migration = False
        
        for item in data:
            if isinstance(item, list):
                if len(item) == 2:
                    # Legacy format (role, text) - extract sources from text
                    role, text = item
                    sources = extract_sources(text) if role == "assistant" else []
                    result.append((role, text, sources))
                    needs_migration = True
                elif len(item) == 3:
                    # New format (role, text, sources) - check if sources need to be extracted
                    role, text, sources = item
                    if role == "assistant" and not sources and ("file://" in text or "`/" in text):
                        # Extract sources from existing assistant messages that don't have sources
                        sources = extract_sources(text)
                        needs_migration = True
                    result.append((role, text, sources))
                else:
                    # Invalid format, skip
                    continue
        
        # Save migrated history if needed
        if needs_migration:
            _save_history(result)
            
        return result
    except Exception:
        return []


def _save_history(history: List[Tuple[str, str, List[str]]]) -> None:
    """Save history with sources."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def clear_history() -> None:
    """Clear chat history."""
    _save_history([])


def _load_ingestion_history() -> List[Dict[str, Any]]:
    """Load ingestion history from file."""
    if not INGESTION_HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(INGESTION_HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_ingestion_history(history: List[Dict[str, Any]]) -> None:
    """Save ingestion history to file."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        INGESTION_HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _add_ingestion_record(job_type: str, details: Dict[str, Any], status: str = "started") -> None:
    """Add a new ingestion record to history."""
    history = _load_ingestion_history()
    record = {
        "id": str(uuid.uuid4()),
        "type": job_type,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details,
        "chunks_before": len(load_index()) if status == "started" else details.get("chunks_before"),
        "chunks_after": details.get("chunks_after"),
        "downloaded": details.get("downloaded", 0),
        "error": details.get("error")
    }
    
    # Keep last 50 records
    history.insert(0, record)
    history = history[:50]
    _save_ingestion_history(history)


def _update_ingestion_record(job_id: str, updates: Dict[str, Any]) -> None:
    """Update an existing ingestion record."""
    history = _load_ingestion_history()
    for record in history:
        if record.get("details", {}).get("job_id") == job_id:
            record.update(updates)
            record["last_update"] = datetime.utcnow().isoformat()
            break
    _save_ingestion_history(history)


# ===============================
# NOTEBOOK HISTORY FUNCTIONS
# ===============================

def _load_notebook_history() -> List[Dict[str, Any]]:
    """Load notebook search history from file."""
    if not NOTEBOOK_HISTORY_FILE.exists():
        return []
    
    try:
        data = json.loads(NOTEBOOK_HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_notebook_history(history: List[Dict[str, Any]]) -> None:
    """Save notebook search history to file."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        NOTEBOOK_HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def save_notebook_search(query: str, root: str, results_count: int, timestamp: str) -> None:
    """Save a notebook search to history."""
    try:
        history = _load_notebook_history()
        
        # Create new search record
        search_record = {
            "query": query,
            "root": root,
            "results_count": results_count,
            "timestamp": timestamp
        }
        
        # Remove duplicate queries (keep only the most recent)
        history = [h for h in history if not (h.get("query") == query and h.get("root") == root)]
        
        # Add new record at the beginning
        history.insert(0, search_record)
        
        # Keep only the last 100 searches
        history = history[:100]
        
        _save_notebook_history(history)
        
    except Exception as e:
        print(f"Error saving notebook search: {e}")


def get_notebook_history() -> List[Dict[str, Any]]:
    """Get notebook search history."""
    return _load_notebook_history()


def clear_notebook_history() -> None:
    """Clear all notebook search history."""
    try:
        if NOTEBOOK_HISTORY_FILE.exists():
            NOTEBOOK_HISTORY_FILE.unlink()
    except Exception:
        pass


def create_agent_from_history() -> Agent:
    system_prompt = (
        "Sei un assistant per knowledge base. Usa kb_search per recuperare contesto e cita le fonti. "
        "Quando l'utente chiede elenchi di file (notebook/slides/python), chiama kb_search con mode='files' e lascia che il filtro estensione sia inferito dalla query. "
        "Per i notebook: usa nb_list per individuarli e nb_report per generare un report HTML riassuntivo SENZA esecuzione. "
        "Restituisci sempre link file:// alle risorse per l'anteprima."
    )
    mem = Memory()
    for role, content, sources in _load_history():
        block = TextBlock(content)
        mem.add_turn(blocks=block, role=ROLE.USER if role == "user" else ROLE.ASSISTANT)


    # Create manual Tool objects to avoid decorator issues
    tools = []
    for func in [kb_search, nb_list, nb_run, nb_report]:
        if hasattr(func, 'name') and hasattr(func, 'description'):
            # It's already a Tool object
            tools.append(func)
        else:
            # Convert function to Tool manually
            tool_obj = Tool(
                name=func.__name__,
                description=func.__doc__ or f"Tool: {func.__name__}",
                func=func
            )
            tools.append(tool_obj)

    agent = Agent(
        name="hb-web-agent",
        system_prompt=system_prompt,
        client=get_chat_client(),
        tools=tools,
        terminate_on_text=True,
        memory=mem,
    )
    return agent


def agent_chat(user_text: str) -> Dict[str, Any]:
    history = _load_history()
    history.append(("user", user_text, []))  # User messages have no sources
    agent = create_agent_from_history()
    res = agent.run(user_text, tool_choice="required_first")
    sources = extract_sources(res)
    history.append(("assistant", res, sources))  # Assistant messages include sources
    _save_history(history)
    return {"reply": res, "history": history, "sources": sources}


def list_index_sources(limit: int = 100) -> List[Dict[str, Any]]:
    rows = load_index()
    out: List[Dict[str, Any]] = []
    seen = set()
    for r in rows:
        src = r.get("metadata", {}).get("source") or r.get("source")
        if src and src not in seen:
            seen.add(src)
            out.append({"source": src, "ext": Path(src).suffix.lower()})
            if len(out) >= limit:
                break
    return out


def clear_history() -> None:
    try:
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        if PREVIEW_FILE.exists():
            PREVIEW_FILE.unlink()
    except Exception:
        pass


def extract_sources(text: str) -> List[str]:
    matches: List[Tuple[int, str]] = []
    for m in re.finditer(r"file://([^\s\)]+)", text):
        p = urllib.parse.unquote(m.group(1))  # type: ignore[attr-defined]
        p = "/" + p.lstrip("/") if not p.startswith("/") else p
        matches.append((m.start(), p))
    for m in re.finditer(r"`(/[^`]+)`", text):
        matches.append((m.start(), m.group(1)))
    matches.sort(key=lambda x: x[0])
    ordered: List[str] = []
    seen: set[str] = set()
    for _, p in matches:
        if p in seen:
            continue
        if Path(p).exists():
            ordered.append(p)
            seen.add(p)
    return ordered


def _ipynb_preview(path: str, max_cells: int, mtime: float) -> List[Tuple[str, str]]:
    import json as _json
    nb = _json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
    cells = nb.get("cells", [])
    out: List[Tuple[str, str]] = []
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


def _pdf_preview(path: str, max_pages: int, zoom: float, mtime: float) -> List[bytes]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    imgs: List[bytes] = []
    with fitz.open(path) as doc:  # type: ignore[attr-defined]
        pages = min(max_pages, len(doc))
        for i in range(pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            imgs.append(pix.tobytes("png"))
    return imgs


def get_preview(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return [{"kind": "error", "message": "File non trovato"}]
    ext = p.suffix.lower()
    try:
        if ext in {".html", ".htm"}:
            html = p.read_text(encoding="utf-8", errors="ignore")
            return [{"kind": "html_inline", "html": html}]
        if ext == ".ipynb":
            preview = _ipynb_preview(str(p), max_cells=8, mtime=os.path.getmtime(p))
            items: List[Dict[str, Any]] = []
            for kind, content in preview:
                if kind == "md":
                    items.append({"kind": "markdown", "text": content})
                elif kind == "code":
                    items.append({"kind": "code", "language": "python", "text": content})
            return items
        if ext == ".py":
            return [{"kind": "code", "language": "python", "text": p.read_text(encoding="utf-8", errors="ignore")}]
        if ext in {".md", ".txt"}:
            return [{"kind": "text", "text": p.read_text(encoding="utf-8", errors="ignore")[:4000]}]
        if ext == ".pdf":
            imgs = _pdf_preview(str(p), max_pages=1, zoom=2.0, mtime=os.path.getmtime(p))
            if imgs:
                import base64
                return [{"kind": "image", "mimetype": "image/png", "data_base64": base64.b64encode(imgs[0]).decode("ascii")}]
            return [{"kind": "info", "message": "Anteprima PDF non disponibile"}]
        if ext in {".pptx", ".docx"}:
            # Fallback: extract plain text through ingest utility if available
            try:
                from demo_minimo.handbook_assistant.ingest import load_text
                return [{"kind": "text", "text": load_text(p)[:4000]}]
            except Exception:
                return [{"kind": "info", "message": "Preview non supportata"}]
        return [{"kind": "info", "message": "Preview non supportata"}]
    except Exception as e:
        return [{"kind": "error", "message": f"Impossibile generare preview: {e}"}]


