"""Interfaz local para el asistente documental de la Unison."""
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = {
    "/": ("index.html", "text/html"),
    "/style.css": ("style.css", "text/css"),
    "/chat.js": ("chat.js", "text/javascript"),
}


class ChatHandler(BaseHTTPRequestHandler):
    pipeline = None

    def send_content(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, data):
        self.send_content(status, json.dumps(data, ensure_ascii=False).encode(), "application/json")

    def do_GET(self):
        asset = ASSETS.get(self.path)
        if asset is None:
            self.send_error(404)
            return
        name, content_type = asset
        self.send_content(200, (ROOT / "web" / name).read_bytes(), content_type)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"error": "La consulta debe enviarse en formato JSON."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16_384:
                raise ValueError
            payload = json.loads(self.rfile.read(length))
            question = payload.get("question") if isinstance(payload, dict) else None
            if not isinstance(question, str) or not 0 < len(question.strip()) <= 2000:
                raise ValueError
        except ValueError:
            self.send_json(400, {"error": "Escribe una pregunta de hasta 2000 caracteres."})
            return

        try:
            if ChatHandler.pipeline is None:
                from src.rag_pipeline_v2 import RAGPipeline

                ChatHandler.pipeline = RAGPipeline(
                    retrieval_mode="semantic_continuation",
                    expand_queries=True,
                    select_passages=True,
                )
            result = ChatHandler.pipeline.ask(question.strip())
            self.send_json(200, {
                "answer": result["answer"],
                "sources": result["sources"],
                "citations": result["citation_check"]["citation_details"],
            })
        except Exception:
            logging.exception("Error al consultar el asistente")
            self.send_json(503, {"error": "No se pudo completar la consulta. Revisa Ollama y el índice de documentos e inténtalo de nuevo."})


if __name__ == "__main__":
    os.chdir(ROOT)
    with HTTPServer(("127.0.0.1", 8765), ChatHandler) as server:
        print("Asistente Unison disponible en http://127.0.0.1:8765", flush=True)
        server.serve_forever()
