# Asistente Unison

Chat local con RAG v2 e índice documental ChromaDB incluido.

## Ejecución en Windows

Instala Python 3.12 u otra versión compatible con las dependencias. Ejecuta Ollama
en el puerto 11434 y usa estos comandos desde la raíz del proyecto.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull bge-m3
ollama pull qwen2.5:3b-instruct
.\.venv\Scripts\python.exe app.py
```

Abre [Asistente Unison](http://127.0.0.1:8765). El índice incluido contiene 1084
fragmentos y conserva los nombres y las páginas de las fuentes.
