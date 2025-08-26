import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import services as svc


BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web_frontend"
REPORTS_DIR = BASE_DIR / "reports"


def create_app() -> FastAPI:
    app = FastAPI(title="Handbook Assistant API", version="0.1.0")

    # CORS: allow from anywhere for dev/gh-pages usage
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=600,
    )

    # Static: reports and web frontend
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if REPORTS_DIR.exists():
        app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR), html=False), name="reports")

    if WEB_DIR.exists():
        # Serve assets under both /assets and /static to support local + gh-pages
        app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets"), html=False), name="assets")
        app.mount("/static", StaticFiles(directory=str(WEB_DIR / "assets"), html=False), name="static")

        @app.get("/", response_class=HTMLResponse)
        def _index() -> Any:
            idx = WEB_DIR / "index.html"
            if not idx.exists():
                return HTMLResponse("<h1>Web frontend non trovato</h1>", status_code=404)
            return HTMLResponse(idx.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True}

    @app.get("/api/reports/list")
    def reports_list() -> Dict[str, Any]:
        files = []
        if REPORTS_DIR.exists():
            for p in REPORTS_DIR.glob("*.html"):
                files.append({"name": p.name, "url": "/reports/" + p.name, "mtime": p.stat().st_mtime})
        files.sort(key=lambda x: x["mtime"], reverse=True)
        return {"items": files}

    @app.get("/api/history")
    def get_history() -> Dict[str, Any]:
        return {"history": svc._load_history()}  # type: ignore[attr-defined]

    @app.post("/api/history/clear")
    def post_clear_history() -> Dict[str, Any]:
        svc.clear_history()
        return {"ok": True}

    @app.get("/api/ingestion/history")
    def get_ingestion_history() -> Dict[str, Any]:
        return {"history": svc._load_ingestion_history()}  # type: ignore[attr-defined]

    @app.post("/api/chat")
    def post_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
        text = (payload or {}).get("text", "")
        if not text:
            raise HTTPException(400, "text richiesto")
        return svc.agent_chat(text)

    @app.get("/api/index/sources")
    def get_sources(limit: int = Query(100, ge=1, le=2000)) -> Dict[str, Any]:
        return {"items": svc.list_index_sources(limit)}

    @app.get("/api/index/summary")
    def get_index_summary() -> Dict[str, Any]:
        rows = svc.load_index()
        return {"chunk_count": len(rows)}

    @app.post("/api/index/clear")
    def post_index_clear() -> Dict[str, Any]:
        idx = svc.INDEX_FILE  # type: ignore[attr-defined]
        try:
            if idx.exists():
                idx.unlink()
            return {"ok": True}
        except Exception as e:
            raise HTTPException(500, f"Errore pulizia indice: {e}")

    @app.get("/api/notebooks")
    def get_notebooks(query: str = "", root: str = "auto", limit: int = 200) -> Dict[str, Any]:
        try:
            items = json.loads(svc.nb_list(query, root, limit))
        except Exception as e:
            raise HTTPException(500, f"Errore lista: {e}")
        return {"items": items}

    @app.post("/api/notebook/report")
    def post_nb_report(payload: Dict[str, Any]) -> Dict[str, Any]:
        path = (payload or {}).get("path")
        max_chars = int((payload or {}).get("max_chars", 12000))
        if not path:
            raise HTTPException(400, "path richiesto")
        res = json.loads(svc.nb_report(path, max_chars))
        if not res.get("ok"):
            raise HTTPException(500, res.get("error", "Errore report"))
        url = None
        if res.get("html_path"):
            p = Path(res["html_path"]).resolve()
            if str(p).startswith(str(REPORTS_DIR.resolve())):
                url = "/reports/" + p.name
        res["url"] = url
        return res



    @app.post("/api/notebook/run")
    def post_nb_run(payload: Dict[str, Any]) -> Dict[str, Any]:
        path = (payload or {}).get("path")
        params_json = (payload or {}).get("params_json")
        if not path:
            raise HTTPException(400, "path richiesto")
        res = json.loads(svc.nb_run(path, params_json))
        if not res.get("ok"):
            raise HTTPException(500, res.get("error", "Errore esecuzione"))
        url = None
        if res.get("html_path"):
            p = Path(res["html_path"]).resolve()
            if str(p).startswith(str(REPORTS_DIR.resolve())):
                url = "/reports/" + p.name
        res["url"] = url
        return res

    @app.get("/api/notebooks/history")
    def get_notebook_history() -> Dict[str, Any]:
        """Get notebook search history."""
        try:
            history = svc.get_notebook_history()
            return {"history": history}
        except Exception as e:
            return {"error": str(e), "history": []}

    @app.post("/api/notebooks/history")
    def post_notebook_history(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Save a notebook search to history."""
        query = (payload or {}).get("query", "")
        root = (payload or {}).get("root", "auto")
        results_count = (payload or {}).get("results_count", 0)
        timestamp = (payload or {}).get("timestamp", "")
        
        if not query or not timestamp:
            raise HTTPException(400, "Query e timestamp richiesti")
        
        try:
            svc.save_notebook_search(query, root, results_count, timestamp)
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    @app.delete("/api/notebooks/history")
    def delete_notebook_history() -> Dict[str, Any]:
        """Clear notebook search history."""
        try:
            svc.clear_notebook_history()
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/preview")
    def get_preview(path: str) -> Dict[str, Any]:
        items = svc.get_preview(path)
        return {"items": items}

    @app.get("/api/file-url")
    def get_file_url(path: str) -> Dict[str, Any]:
        p = Path(path).resolve()
        url = None
        if str(p).startswith(str(REPORTS_DIR.resolve())):
            url = "/reports/" + p.name
        return {"url": url}



    @app.post("/api/ingest/local")
    def post_ingest_local(payload: Dict[str, Any]) -> Dict[str, Any]:
        subdir = (payload or {}).get("subdir")
        try:
            from demo_minimo.handbook_assistant.ingest import main as ingest_main

            ingest_main(subdir=subdir or None)
            return {"ok": True}
        except Exception as e:
            raise HTTPException(500, f"Errore ingestion: {e}")

    @app.post("/api/ingest/drive")
    def post_ingest_drive(payload: Dict[str, Any]) -> Dict[str, Any]:
        urls: List[str] = [u.strip() for u in (payload or {}).get("urls", []) if u and u.strip()]
        if not urls:
            raise HTTPException(400, "urls richiesto")
        try:
            job_id = svc.start_drive_ingest(urls)
            return {"ok": True, "job_id": job_id}
        except Exception as e:
            raise HTTPException(500, f"Errore Drive: {e}")

    @app.post("/api/ingest/drive/test")
    def post_ingest_drive_test(payload: Dict[str, Any]) -> Dict[str, Any]:
        urls: List[str] = [u.strip() for u in (payload or {}).get("urls", []) if u and u.strip()]
        if not urls:
            raise HTTPException(400, "urls richiesto")
        try:
            svc.ensure_drive_sa_env()
            # Minimal list of first folder children to verify access
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            import re
            # Extract folder id
            url_or_id = urls[0]
            fid = url_or_id
            if "/folders/" in url_or_id:
                fid = url_or_id.split("/folders/")[-1].split("?")[0]
            elif "/drive/folders/" in url_or_id:
                fid = url_or_id.split("/drive/folders/")[-1].split("?")[0]

            cred_path = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
            if not cred_path or not Path(cred_path).exists():
                raise RuntimeError("Service account JSON non configurato")
            scopes = ["https://www.googleapis.com/auth/drive.readonly"]
            creds = service_account.Credentials.from_service_account_file(cred_path, scopes=scopes)
            service = build("drive", "v3", credentials=creds)
            res = service.files().list(q=f"'{fid}' in parents and trashed=false", fields="files(id,name,mimeType)").execute()
            files = res.get("files", [])
            return {"ok": True, "count": len(files), "sample": files[:10]}
        except Exception as e:
            raise HTTPException(500, f"Errore test Drive: {e}")

    @app.get("/api/ingest/state")
    def get_ingest_state(job_id: Optional[str] = None) -> Dict[str, Any]:
        return svc.get_ingest_state(job_id)

    @app.post("/api/ingest/start-job")
    def post_start_ingestion_job(payload: Dict[str, Any]) -> Dict[str, Any]:
        job_type = (payload or {}).get("type", "")
        if not job_type:
            raise HTTPException(400, "type richiesto (drive|local)")
        
        try:
            if job_type == "drive":
                urls: List[str] = [u.strip() for u in (payload or {}).get("urls", []) if u and u.strip()]
                if not urls:
                    raise HTTPException(400, "urls richiesto per type=drive")
                job_id = svc.start_drive_ingest(urls)
                return {"ok": True, "job_id": job_id}
            elif job_type == "local":
                subdir = (payload or {}).get("subdir")
                job_id = svc.start_local_ingest(subdir)
                return {"ok": True, "job_id": job_id}
            else:
                raise HTTPException(400, f"Tipo job non supportato: {job_type}")
        except Exception as e:
            raise HTTPException(500, f"Errore avvio job: {e}")

    @app.post("/api/drive/set-service-account")
    def post_set_service_account(payload: Dict[str, Any]) -> Dict[str, Any]:
        content = (payload or {}).get("content")
        if content is None:
            raise HTTPException(400, "content richiesto (JSON oggetto o stringa)")
        try:
            if isinstance(content, str):
                obj = json.loads(content)
            else:
                obj = content
            if not isinstance(obj, dict):
                raise ValueError("Il contenuto deve essere un oggetto JSON")
            svc.STATE_DIR.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
            sa_path = svc.STATE_DIR / "drive_sa.json"  # type: ignore[attr-defined]
            sa_path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
            os.environ["GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"] = str(sa_path)
            return {"ok": True}
        except Exception as e:
            raise HTTPException(500, f"Errore salvataggio service account: {e}")

    @app.post("/api/ingest/discord")
    def post_ingest_discord(payload: Dict[str, Any]) -> Dict[str, Any]:
        token = (payload or {}).get("bot_token") or ""
        ch_text = (payload or {}).get("channel_ids") or []
        max_msgs = int((payload or {}).get("max_messages", 500))
        if not token:
            raise HTTPException(400, "bot_token richiesto")
        try:
            from demo_minimo.handbook_assistant.discord_ingest import ingest_discord

            added = ingest_discord(token, ch_text, max_messages_per_channel=max_msgs)
            return {"ok": True, "indexed": added}
        except Exception as e:
            raise HTTPException(500, f"Errore Discord: {e}")

    @app.post("/api/share/ngrok")
    def post_share_ngrok(payload: Dict[str, Any]) -> Dict[str, Any]:
        token = (payload or {}).get("token") or os.getenv("NGROK_AUTHTOKEN", "")
        try:
            from pyngrok import ngrok

            if token:
                try:
                    ngrok.set_auth_token(token)
                except Exception:
                    pass
            # Prefer HTTP 8000 by default
            port = int(os.getenv("PORT", "8000"))
            url = ngrok.connect(addr=port, proto="http").public_url
            return {"ok": True, "url": url}
        except Exception as e:
            raise HTTPException(500, f"Errore ngrok: {e}")

    @app.get("/api/config")
    def get_config() -> Dict[str, Any]:
        info = svc.config_state()
        return {"reports_base": "/reports/", "api_base": "", **info}

    @app.post("/api/config/openai-key")
    def set_openai_key(payload: Dict[str, Any]) -> Dict[str, Any]:
        key = (payload or {}).get("api_key", "").strip()
        if not key:
            raise HTTPException(400, "api_key richiesto")
        try:
            svc.set_openai_api_key(key)
            return {"ok": True}
        except Exception as e:
            raise HTTPException(500, f"Errore set API key: {e}")

    @app.post("/api/report/ai")
    def post_generic_ai_report(payload: Dict[str, Any]) -> Dict[str, Any]:
        path = (payload or {}).get("path")
        max_chars = int((payload or {}).get("max_chars", 12000))
        if not path:
            raise HTTPException(400, "path richiesto")
        res = json.loads(svc.ai_report_for_file(path, max_chars))
        if not res.get("ok"):
            raise HTTPException(500, res.get("error", "Errore report AI"))
        
        # Add URL if html_path exists
        if res.get("html_path"):
            html_path = Path(res["html_path"])
            if html_path.exists():
                res["url"] = f"/reports/{html_path.name}"
        
        return res

    @app.get("/api/file/download")
    def download_file(path: str):
        p = Path(path).resolve()
        if not p.exists() or not p.is_file():
            raise HTTPException(404, "File non trovato")
        # Allow access to data directory and reports
        data_dir = (BASE_DIR / "data").resolve()
        reports_dir = (BASE_DIR / "reports").resolve()
        remote_cache_dir = (BASE_DIR / "remote_cache").resolve()
        
        allowed_paths = [BASE_DIR.resolve(), data_dir, reports_dir, remote_cache_dir]
        is_allowed = any(str(p).startswith(str(allowed_path)) for allowed_path in allowed_paths)
        
        if not is_allowed:
            raise HTTPException(403, "Accesso negato")
        return FileResponse(str(p), filename=p.name)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("demo_minimo.handbook_assistant.api.server:app", host="0.0.0.0", port=port, reload=False)


