
"""
RAG Pipeline V5.1 - Asistente Virtual UNISON

Arquitectura:
    PDF -> ChromaDB -> BGE-M3 -> Recuperación
    -> Reranking -> Qwen 2.5 -> Validación de evidencia

Características:
    - Recuperación semántica
    - Recuperación híbrida opcional
    - Búsqueda léxica complementaria
    - Reranking BM25 ligero
    - Generación estructurada JSON
    - Validación de citas textuales
    - Abstención cuando falta evidencia
    - Diagnóstico de errores de validación
    - Métricas de tiempo de ejecución
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import unicodedata

from collections import Counter
from typing import Any

import requests

from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

ABSTENTION = (
    "No encontré información suficiente en los documentos "
    "proporcionados para responder esta pregunta."
)

RETRIEVAL_MODES = (
    "semantic", "hybrid", "hybrid_reranked", "mixed", "semantic_continuation",
)

STOPWORDS = {
    "que", "para", "como", "cual", "cuales",
    "cuanto", "cuantos", "cuanta", "cuantas",
    "debo", "puedo", "tengo", "una", "uno",
    "los", "las", "del", "por", "con", "sin",
    "sobre", "esta", "este", "son", "segun",
    "universidad", "sonora", "reglamento",
    "necesito", "de", "el", "la", "en",
    "un", "y", "a", "se", "al", "mi",
    "es", "hay", "ser", "tener"
}


# ============================================================
# NORMALIZACIÓN DE TEXTO
# ============================================================

def normalize_text(text: str) -> str:

    text = unicodedata.normalize(
        "NFKD",
        str(text).casefold()
    )

    return "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )


def tokenize(text: str) -> list[str]:

    normalized = normalize_text(text)

    words = re.findall(
        r"[a-z0-9]+",
        normalized
    )

    return [
        word
        for word in words
        if len(word) > 2
        and word not in STOPWORDS
    ]


def fee_terms(text: str) -> set[str]:
    """Normaliza equivalencias de vocabulario para buscar servicios tarifados."""
    terms = set(tokenize(text))
    if "reposicion" in terms:
        terms.remove("reposicion")
        terms.add("duplicado")
    return terms


# ============================================================
# VALIDACIÓN DE CITAS TEXTUALES
# ============================================================

def canonical_text(text: str) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "",
        normalize_text(text)
    )


def quote_supported(
    quote: str,
    document_text: str
) -> bool:

    quote_normalized = canonical_text(quote)

    document_normalized = canonical_text(
        document_text
    )

    if len(quote_normalized) < 12:
        return False

    return quote_normalized in document_normalized


# ============================================================
# BM25
# ============================================================

def bm25_scores(
    query: str,
    documents: list[dict]
) -> list[float]:

    corpus = [
        tokenize(doc.get("text", ""))
        for doc in documents
    ]

    query_terms = set(tokenize(query))

    if not corpus or not query_terms:
        return [0.0] * len(documents)

    n_documents = len(corpus)

    average_length = max(
        sum(len(doc) for doc in corpus) / n_documents,
        1.0
    )

    document_frequency = Counter()

    for document in corpus:
        document_frequency.update(set(document))

    scores = []

    k1 = 1.2
    b = 0.75

    for document in corpus:

        term_frequency = Counter(document)

        document_length = len(document)

        score = 0.0

        for term in query_terms:

            frequency = term_frequency[term]

            if frequency == 0:
                continue

            df = document_frequency[term]

            idf = math.log(
                1 + (
                    n_documents - df + 0.5
                ) / (
                    df + 0.5
                )
            )

            numerator = (
                frequency * (k1 + 1)
            )

            denominator = (
                frequency
                + k1 * (
                    1 - b
                    + b * document_length
                    / average_length
                )
            )

            score += (
                idf * numerator / denominator
            )

        scores.append(score)

    return scores


# ============================================================
# PUNTUACIÓN DE EVIDENCIA
# ============================================================

def evidence_score(
    question: str,
    document: dict
) -> float:

    question_terms = set(tokenize(question))

    if not question_terms:
        return 0.0

    text = document.get("text", "")

    sentences = re.split(
        r"(?<=[.;:])\s+|\n+",
        text
    )

    best_score = 0.0

    for sentence in sentences:

        sentence_terms = set(
            tokenize(sentence)
        )

        overlap = len(
            question_terms & sentence_terms
        )

        score = overlap / len(question_terms)

        best_score = max(
            best_score,
            score
        )

    return best_score


# ============================================================
# CLASE PRINCIPAL
# ============================================================

class RAGPipeline:

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        ollama_url: str = "http://localhost:11434",
        top_k: int = 5,
        candidate_k: int = 20,
        retrieval_mode: str = "semantic",
        collection_name: str = "unison_documents_v2",
        semantic_weight: float = 0.8,
        bm25_weight: float = 0.2,
        timeout: int = 180,
        expand_queries: bool = False,
        select_passages: bool = False,
    ):

        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(
                "retrieval_mode no reconocido"
            )

        if top_k < 1 or candidate_k < top_k:
            raise ValueError(
                "candidate_k debe ser >= top_k >= 1"
            )

        self.model = model

        self.ollama_url = ollama_url.rstrip("/")

        self.top_k = top_k
        self.candidate_k = candidate_k

        self.retrieval_mode = retrieval_mode

        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        self.timeout = timeout
        self.expand_queries = expand_queries
        self.select_passages = select_passages

        # Base vectorial
        self.vector_store = VectorStore(
            persist_directory="data/chroma_db",
            collection_name=collection_name
        )

        document_count = self.vector_store.count()

        if document_count == 0:
            raise RuntimeError(
                "La colección de ChromaDB está vacía."
            )

        logger.info(
            "ChromaDB: %s fragmentos",
            document_count
        )

        # Modelo de embeddings
        self.embedder = EmbeddingGenerator(
            model="bge-m3",
            base_url=self.ollama_url
        )

        # Caché para búsqueda léxica
        self._lexical_cache = None

    # ========================================================
    # OBTENER DOCUMENTOS DE CHROMADB
    # ========================================================

    def _lexical_documents(self) -> list[dict]:

        if self._lexical_cache is not None:
            return self._lexical_cache

        documents = []

        offset = 0

        while True:

            batch = self.vector_store.collection.get(
                limit=500,
                offset=offset,
                include=[
                    "documents",
                    "metadatas"
                ]
            )

            ids = batch.get("ids") or []

            if not ids:
                break

            texts = batch.get("documents") or []
            metadatas = batch.get("metadatas") or []

            for index, document_id in enumerate(ids):

                documents.append({
                    "id": document_id,
                    "text": texts[index],
                    "metadata": metadatas[index] or {},
                    "distance": None
                })

            offset += len(ids)

            if len(ids) < 500:
                break

        self._lexical_cache = documents

        return documents

    # ========================================================
    # RECUPERACIÓN SEMÁNTICA
    # ========================================================

    def expand_question(self, question: str) -> list[str]:
        """Genera hasta dos variantes para la búsqueda documental."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "keep_alive": "30m",
                    "options": {"temperature": 0, "num_predict": 256},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                'Devuelve un objeto JSON con la clave "queries" y una lista de '
                                "hasta dos reformulaciones breves para buscar en documentos de "
                                "la Universidad de Sonora. Usa terminología institucional y "
                                "conserva la intención, los nombres, las fechas y las condiciones "
                                "de la pregunta. No respondas ni inventes detalles. Interpreta "
                                "expresiones coloquiales cuando el tema sea reconocible. Ejemplo: "
                                'para "¿Puedo dejar una materia?" o "¿Cómo suelto una clase?", devuelve '
                                '{"queries":["baja de asignatura","procedimiento para dar de baja '
                                'una materia"]}. Si falta el tema, como en "¿Cuándo la puedo '
                                'pedir?", devuelve {"queries":[]}.'
                            ),
                        },
                        {"role": "user", "content": question},
                    ],
                },
                timeout=min(self.timeout, 30),
            )
            response.raise_for_status()
            payload = json.loads(response.json()["message"]["content"])
            variants = payload.get("queries") if isinstance(payload, dict) else None
            if not isinstance(variants, list):
                return []
        except (requests.RequestException, KeyError, TypeError, ValueError):
            logger.warning("No se pudo expandir la consulta; se usará la pregunta original.")
            return []

        seen = {" ".join(question.casefold().split())}
        queries = []
        for variant in variants:
            if not isinstance(variant, str):
                continue
            variant = " ".join(variant.split())
            key = variant.casefold()
            if not variant or len(variant) > 500 or key in seen:
                continue
            queries.append(variant)
            seen.add(key)
            if len(queries) == 2:
                break
        return queries

    def retrieve_context(self, question: str) -> list[dict]:
        results = self._retrieve_ranked(question)
        if self.retrieval_mode == "semantic_continuation":
            return self._include_continuations(results)
        return results

    def _include_continuations(self, results: list[dict]) -> list[dict]:
        """Reúne un encabezado de lista con el fragmento que lo continúa."""
        if not any(doc["text"].rstrip().endswith(":") for doc in results):
            return results
        by_position = {}
        for doc in self._lexical_documents():
            meta = doc["metadata"]
            if all(key in meta for key in ("source", "page", "chunk_index")):
                by_position[(meta["source"], meta["page"], meta["chunk_index"])] = doc
        ordered = {}
        for doc in results:
            ordered.setdefault(doc["id"], doc)
            text = doc["text"].rstrip()
            metadata = doc["metadata"]
            index = metadata.get("chunk_index")
            if not text.endswith(":") or not isinstance(index, int):
                continue
            successor = by_position.get((metadata.get("source"), metadata.get("page"), index + 1))
            if successor is None:
                continue
            intro = " ".join(text.split())[-60:]
            following = " ".join(successor["text"].split())
            if intro in following[:240] or re.match(r"^(?:[IVX]+[.)]|\d+[.)]|[•-])\s", following):
                ordered.setdefault(successor["id"], successor)
        return list(ordered.values())[:self.top_k]

    def _retrieve_ranked(self, question: str) -> list[dict]:
        variants = self.expand_question(question) if self.expand_queries else []
        original = self._search_question(question)
        if not variants or self.top_k == 1:
            return original

        documents = {}
        scores = {}
        for index, query in enumerate([question, *variants]):
            if index == 0:
                results = original
            else:
                try:
                    results = self._search_question(query)
                except (requests.RequestException, ValueError):
                    logger.warning("No se pudo recuperar contexto para una reformulación.")
                    continue
            weight = 1.0 if index == 0 else 0.75
            for rank, document in enumerate(results, start=1):
                key = document["id"]
                documents.setdefault(key, document)
                scores[key] = scores.get(key, 0) + weight / (60 + rank)

        ranked = sorted(documents, key=scores.__getitem__, reverse=True)
        return [documents[key] for key in ranked[:self.top_k]]

    def _search_question(
        self,
        question: str
    ) -> list[dict]:

        embedding = self.embedder.generate_embeddings(
            [question]
        )[0]

        semantic_results = self.vector_store.search(
            query_embedding=embedding,
            n_results=self.candidate_k
        )

        if not semantic_results:
            return []

        hybrid_ranked = []
        if self.retrieval_mode in ("hybrid", "hybrid_reranked", "mixed"):

            corpus = self._lexical_documents()

            lexical_scores = bm25_scores(
                question,
                corpus
            )

            ranked_indices = sorted(
                range(len(corpus)),
                key=lambda i: (
                    -lexical_scores[i],
                    i
                )
            )

            lexical_results = [
                corpus[i]
                for i in ranked_indices
                if lexical_scores[i] > 0
            ][:self.candidate_k]

            merged = {
                doc["id"]: dict(doc)
                for doc in semantic_results
            }

            for doc in lexical_results:
                merged.setdefault(
                    doc["id"],
                    dict(doc)
                )

            rrf_scores = Counter()

            for rank, doc in enumerate(
                semantic_results,
                start=1
            ):

                rrf_scores[doc["id"]] += (
                    self.semantic_weight
                    / (60 + rank)
                )

            for rank, doc in enumerate(
                lexical_results,
                start=1
            ):

                rrf_scores[doc["id"]] += (
                    self.bm25_weight
                    / (60 + rank)
                )

            if self.retrieval_mode == "hybrid_reranked":
                query_terms = set(tokenize(question))
                for doc in merged.values():
                    text_terms = set(tokenize(doc["text"]))
                    title_terms = set(tokenize(doc["metadata"].get("source", "")))
                    coverage = len(query_terms & text_terms) / max(len(query_terms), 1)
                    title_match = len(query_terms & title_terms) / max(len(query_terms), 1)
                    rrf_scores[doc["id"]] += (
                        0.006 * coverage
                        + 0.004 * evidence_score(question, doc)
                        + 0.002 * title_match
                    )

            hybrid_ranked = sorted(merged.values(), key=lambda doc: -rrf_scores[doc["id"]])
            if self.retrieval_mode != "mixed":
                return hybrid_ranked[:self.top_k]

        # Reranking semántico
        lexical_scores = bm25_scores(
            question,
            semantic_results
        )

        max_lexical = (
            max(lexical_scores, default=0.0)
            or 1.0
        )

        ranked = []

        for index, doc in enumerate(
            semantic_results
        ):

            semantic_rank_score = (
                1.0 / (index + 2)
            )

            lexical_score = (
                lexical_scores[index]
                / max_lexical
            )

            local_evidence_score = evidence_score(
                question,
                doc
            )

            score = (
                semantic_rank_score
                + 0.12 * lexical_score
                + 0.20 * local_evidence_score
            )

            ranked.append((
                score,
                index,
                doc
            ))

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1]
            )
        )

        semantic_ranked = [doc for _, _, doc in ranked]
        if not hybrid_ranked:
            return semantic_ranked[:self.top_k]

        # Conserva la mayoría de los resultados semánticos y reserva
        # posiciones para evidencia que solo recuperó la búsqueda léxica.
        semantic_slots = max(1, (self.top_k + 1) // 2)
        selected = {
            doc["id"]: doc
            for doc in [*semantic_ranked[:semantic_slots], *hybrid_ranked, *semantic_ranked]
        }
        return list(selected.values())[:self.top_k]

    # ========================================================
    # RECUPERACIÓN COMPLEMENTARIA
    # ========================================================

    def retrieve_definition(self, question: str) -> dict | None:
        """Encuentra una definición explícita y completa en un artículo."""
        match = re.fullmatch(
            r"(?:quienes son|que son|quien es|que es)\s+(.+)",
            normalize_text(question).strip(" ¿?"),
        )
        if not match:
            return None

        subject = re.sub(r"^(?:los|las|el|la|un|una)\s+", "", match[1])
        subject = re.sub(r"\bestudiante(s?)\b", r"alumno\1", subject)
        article_header = re.compile(
            r"^[ \t]*ART[IÍ]CULO\s+\d+(?:-[A-Z]+)?[.\s-]+",
            re.IGNORECASE | re.MULTILINE,
        )
        definitions = {}
        for doc in self._lexical_documents():
            text = doc["text"]
            headers = list(article_header.finditer(text))
            for index, header in enumerate(headers):
                end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
                quote = text[header.start():end].strip()
                quote = re.sub(r"\n\s*\d+\s*$", "", quote).strip()
                body = " ".join(normalize_text(text[header.end():end]).split())
                if not re.match(
                    rf"(?:son|es)\s+(?:(?:los|las|el|la)\s+)?{re.escape(subject)}\s*:",
                    body,
                ):
                    continue
                if not quote.endswith(".") or len(quote) > 1200:
                    continue
                metadata = doc["metadata"]
                key = (metadata.get("source"), metadata.get("page"), canonical_text(quote))
                definitions[key] = dict(doc, definition_quote=quote)

        return next(iter(definitions.values())) if len(definitions) == 1 else None

    def retrieve_fee_context(self, question: str) -> list[dict]:
        """Busca filas tarifarias por servicio y conserva su fragmento original."""
        normalized = normalize_text(question)
        if not re.search(
            r"\b(cuesta[n]?|costos?|precio|cuotas?|tarifa|pagar|paga|uma)\b",
            normalized,
        ):
            return []

        terms = fee_terms(normalized) - {
            "cuesta", "cuestan", "costo", "costos", "precio", "cuota",
            "cuotas", "tarifa", "pagar", "paga", "uma", "tramite",
        }
        if {"titulo", "licenciatura"} <= terms:
            terms = (terms - {"licenciatura"}) | {"profesional"}
        if not terms:
            return []

        fee_row = re.compile(
            r"\s*(.+?)\s+(\d+(?:[.,]\d+)?\s+uma|exent[oa])\s*",
            re.IGNORECASE,
        )
        ranked = []
        exact_rows = {}
        for doc in self._lexical_documents():
            best = 0
            for line in doc["text"].splitlines():
                match = fee_row.fullmatch(line)
                if match:
                    overlap = terms & fee_terms(match[1])
                    if len(overlap) >= min(2, len(terms)):
                        best = max(best, len(overlap))
                    if len(terms) >= 2 and overlap == terms:
                        metadata = doc["metadata"]
                        key = (metadata.get("source"), metadata.get("page"), line.strip())
                        exact_rows.setdefault(key, (doc, line.strip(), match[1], match[2]))
            if best:
                ranked.append((best, doc))

        ranked.sort(key=lambda item: -item[0])
        selected = {doc["id"]: doc for _, doc in ranked[:2]}
        if len(exact_rows) == 1:
            doc, quote, service, fee = next(iter(exact_rows.values()))
            exact = dict(doc, fee_evidence=(quote, service, fee))
            selected.pop(doc["id"], None)
            return [exact, *selected.values()][:2]
        return list(selected.values())

    def retrieve_targeted_context(
        self,
        question: str,
        current_results: list[dict],
        fee_results: list[dict] | None = None,
    ) -> list[dict]:

        normalized_question = normalize_text(
            question
        )

        if re.search(r"\bdonde\b", normalized_question) and "servicio social" in normalized_question:
            placements = next((
                doc for doc in self._lexical_documents()
                if "servicio social" in normalize_text(doc["metadata"].get("source", ""))
                and re.search(r"(?m)^articulo\s+24\.", normalize_text(doc["text"]))
            ), None)
            if placements:
                selected = {placements["id"]: placements}
                for doc in current_results:
                    selected.setdefault(doc["id"], doc)
                return list(selected.values())[:self.top_k]

        if (
            "carga academica" in normalized_question
            and re.search(r"\bposgrados?\b", normalized_question)
        ):
            criteria = [
                doc for doc in self._lexical_documents()
                if doc["metadata"].get("source") == "criterios_programacionacad_posgrados.pdf"
            ]
            criteria.sort(key=lambda doc: (
                doc["metadata"].get("page", 0),
                doc["metadata"].get("chunk_index", 0),
            ))
            selected = {doc["id"]: doc for doc in criteria}
            for doc in current_results:
                selected.setdefault(doc["id"], doc)
            return list(selected.values())[:self.top_k]

        if fee_results is None:
            fee_results = self.retrieve_fee_context(question)
        if fee_results:
            selected = {doc["id"]: doc for doc in fee_results}
            for doc in current_results:
                selected.setdefault(doc["id"], doc)
            return list(selected.values())[:self.top_k]

        target_terms = []

        if "credencial" in normalized_question:

            target_terms = [
                "credencial",
                "reposicion"
            ]

        elif (
            "rectoria" in normalized_question
            or "rector" in normalized_question
        ):

            target_terms = [
                "rector",
                "colegio universitario",
                "eleccion"
            ]

        elif (
            "posgrado" in normalized_question
            and "calificacion" in normalized_question
        ):

            target_terms = [
                "calificacion minima aprobatoria",
                "ochenta"
            ]

        if not target_terms:
            return current_results

        corpus = self._lexical_documents()

        scored = []

        for doc in corpus:

            text = normalize_text(
                doc.get("text", "")
            )

            score = sum(
                1
                for term in target_terms
                if term in text
            )

            if score > 0:

                scored.append((
                    score,
                    doc
                ))

        scored.sort(
            key=lambda item: -item[0]
        )

        # Reservar dos posiciones para evidencia dirigida
        selected = current_results[
            :max(self.top_k - 2, 0)
        ]

        selected_ids = {
            doc["id"]
            for doc in selected
        }

        for _, doc in scored:

            if doc["id"] in selected_ids:
                continue

            selected.append(doc)

            selected_ids.add(doc["id"])

            if len(selected) >= self.top_k:
                break

        # Completar con recuperación semántica
        for doc in current_results:

            if len(selected) >= self.top_k:
                break

            if doc["id"] in selected_ids:
                continue

            selected.append(doc)

            selected_ids.add(doc["id"])

        return selected[:self.top_k]

    # ========================================================
    # CONSTRUIR CONTEXTO
    # ========================================================

    def build_context(
        self,
        results: list[dict]
    ) -> str:

        sections = []

        for index, doc in enumerate(
            results,
            start=1
        ):

            metadata = doc.get("metadata") or {}

            source = metadata.get(
                "source",
                "Desconocido"
            )

            page = metadata.get(
                "page",
                "Desconocida"
            )

            text = doc.get("text", "")

            section = (
                f"[ID S{index}]\n"
                f"Documento: {source}\n"
                f"Página del PDF: {page}\n"
                f"Texto:\n{text}"
            )

            sections.append(section)

        return "\n\n".join(sections)

    # ========================================================
    # GENERACIÓN
    # ========================================================

    def generate_answer(
        self,
        question: str,
        context: str,
        evidence_ids: list[str] | None = None,
    ) -> tuple[dict, dict]:

        system_prompt = f"""
Eres un asistente documental de la Universidad de Sonora.

Responde exclusivamente con la información proporcionada
en los fragmentos documentales.

REGLAS:

1. No inventes información.

2. Identifica el artículo o disposición que responde
directamente la pregunta.

3. Distingue licenciatura y posgrado.

4. Distingue cantidades de alumnos, créditos, horas,
meses, años y calificaciones.

5. No confundas la duración de un programa con el
plazo para obtener el grado académico.

6. Si existen disposiciones distintas según el nivel
académico, explica la diferencia.

7. No atribuyas facultades de una autoridad a otra.

8. Utiliza máximo dos fuentes directamente pertinentes.

9. Usa los identificadores S que aparecen junto a cada fragmento.
"""

        literal_quote_prompt = f"""
10. En cada evidencia, copia en cita una secuencia literal del Texto
de la fuente elegida que sustente la respuesta. Conserva incluso los
errores de transcripción en la cita. Redacta la respuesta con ortografía normal.

11. No escribas referencias dentro de la respuesta.

12. Si no existe evidencia suficiente, abstente.

13. No respondas usando conocimientos externos.

14. Para cuotas, busca la fila del servicio solicitado y cita
el nombre del servicio junto con el importe y su unidad. En una
tabla, «título profesional» corresponde al título de licenciatura.
Una fila que diga «título profesional o de posgrado» cubre ambos.
La expedición del título y el examen profesional son servicios distintos.

15. Para la asignación de carga académica en posgrados, distingue
los programas que pertenecen al PNP de los que no pertenecen
si los documentos contienen ambos apartados.

Devuelve un objeto JSON con evidencia_suficiente (booleano),
respuesta (texto) y evidencias (lista de objetos con id y cita).

Cuando no exista evidencia:

{{
    "evidencia_suficiente": false,
    "respuesta": "{ABSTENTION}",
    "evidencias": []
}}

Devuelve exclusivamente JSON válido.
"""

        output_format = "json"
        if evidence_ids:
            system_prompt += (
                "Devuelve un objeto JSON con evidencia_suficiente (booleano), "
                "respuesta (texto) y evidencias (lista de identificadores S de los "
                "pasajes que sustentan la respuesta). Usa solo los pasajes dados. "
                "Separa las categorías o supuestos que distingue la fuente y "
                "selecciona evidencia para cada afirmación principal. Cada criterio "
                "pertenece al apartado donde aparece y solo aplica a los demás si "
                "el documento lo afirma expresamente. Para preguntas generales, "
                "resume el procedimiento central de cada apartado y omite detalles "
                "que no sean necesarios para contestar. "
                "Usa el texto de respuesta para explicar, y deja las referencias "
                "en el campo evidencias. "
                "Si los pasajes no sustentan una respuesta, indica "
                "evidencia_suficiente false y devuelve evidencias vacías."
            )
            output_format = {
                "type": "object",
                "properties": {
                    "evidencia_suficiente": {"type": "boolean"},
                    "respuesta": {"type": "string"},
                    "evidencias": {
                        "type": "array",
                        "items": {"type": "string", "enum": evidence_ids},
                    },
                },
                "required": ["evidencia_suficiente", "respuesta", "evidencias"],
            }
        else:
            system_prompt += literal_quote_prompt

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": (
                        "DOCUMENTOS:\n"
                        f"{context}\n\n"
                        "PREGUNTA:\n"
                        f"{question}"
                    )
                }
            ],
            "format": output_format,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192
            }
        }

        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()

        data = response.json()

        raw_answer = data[
            "message"
        ]["content"].strip()

        try:

            parsed = json.loads(
                raw_answer
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "El modelo no devolvió JSON válido: "
                + raw_answer[:300]
            ) from error

        if not isinstance(parsed, dict):
            raise ValueError(
                "La respuesta no es un objeto JSON."
            )

        required_fields = [
            "evidencia_suficiente",
            "respuesta",
            "evidencias"
        ]

        for field in required_fields:

            if field not in parsed:

                raise ValueError(
                    f"Falta el campo: {field}"
                )

        if not isinstance(
            parsed["evidencia_suficiente"],
            bool
        ):
            raise ValueError(
                "evidencia_suficiente debe ser booleano."
            )

        if not isinstance(
            parsed["respuesta"],
            str
        ):
            raise ValueError(
                "respuesta debe ser texto."
            )

        if not isinstance(
            parsed["evidencias"],
            list
        ):
            raise ValueError(
                "evidencias debe ser una lista."
            )

        ns = 1e9

        output_duration = (
            data.get("eval_duration", 0)
            / ns
        )

        output_tokens = data.get(
            "eval_count",
            0
        )

        stats = {
            "model_load_s": (
                data.get("load_duration", 0)
                / ns
            ),
            "prompt_tokens": data.get(
                "prompt_eval_count",
                0
            ),
            "prompt_eval_s": (
                data.get(
                    "prompt_eval_duration",
                    0
                ) / ns
            ),
            "output_tokens": output_tokens,
            "output_eval_s": output_duration,
            "tokens_per_s": (
                output_tokens / output_duration
                if output_duration > 0
                else 0
            ),
            "ollama_total_s": (
                data.get("total_duration", 0)
                / ns
            )
        }

        return parsed, stats

    # ========================================================
    # CONSULTA COMPLETA
    # ========================================================

    def ask(
        self,
        question: str
    ) -> dict[str, Any]:

        if not question or not question.strip():
            raise ValueError(
                "La pregunta no puede estar vacía."
            )

        start = time.perf_counter()

        # ----------------------------------------------------
        # Recuperación
        # ----------------------------------------------------

        retrieval_start = time.perf_counter()

        definition = self.retrieve_definition(question)
        if definition:
            results = [definition]
        else:
            fee_results = self.retrieve_fee_context(question)
            if fee_results and fee_results[0].get("fee_evidence"):
                results = fee_results[:1]
            else:
                results = self.retrieve_targeted_context(
                    question, self.retrieve_context(question), fee_results=fee_results
                )

        retrieval_s = (
            time.perf_counter()
            - retrieval_start
        )

        # ----------------------------------------------------
        # Fuentes recuperadas
        # ----------------------------------------------------

        sources = []

        for doc in results:

            metadata = doc.get("metadata") or {}

            source_info = {
                "document": metadata.get("source"),
                "page": metadata.get("page")
            }

            if source_info not in sources:
                sources.append(source_info)

        # ----------------------------------------------------
        # Sin documentos
        # ----------------------------------------------------

        if not results:

            return {
                "answer": ABSTENTION,
                "sources": [],
                "retrieved_documents": [],
                "retrieval_mode": self.retrieval_mode,
                "citation_check": {
                    "valid_source_ids": [],
                    "invalid_source_ids": [],
                    "has_valid_source_citation": False,
                    "citation_details": [],
                    "citation_count": 0,
                    "model_abstained": True,
                    "validation_failed": False
                },
                "timings": {
                    "retrieval_s": retrieval_s,
                    "generation_s": 0.0,
                    "total_s": (
                        time.perf_counter()
                        - start
                    )
                }
            }

        # ----------------------------------------------------
        # Generación
        # ----------------------------------------------------

        generation_start = time.perf_counter()

        fee_evidence = results[0].get("fee_evidence")
        definition_quote = results[0].get("definition_quote")
        if definition_quote:
            parsed = {
                "respuesta": definition_quote,
                "evidencia_suficiente": True,
                "evidencias": [{"id": "S1", "cita": definition_quote}],
            }
            stats = {}
        elif fee_evidence:
            quote, service, fee = fee_evidence
            parsed = {
                "respuesta": f"{service} tiene una cuota de {fee}." if fee[0].isdigit()
                else f"El servicio «{service}» está exento de cuota.",
                "evidencia_suficiente": True,
                "evidencias": [{"id": "S1", "cita": quote}],
            }
            stats = {}
        else:
            context = self.build_context(results)
            if self.select_passages or all(
                doc["metadata"].get("source") == "criterios_programacionacad_posgrados.pdf"
                for doc in results
            ):
                passages = {
                    f"S{index}": doc["text"]
                    for index, doc in enumerate(results, 1)
                }
                parsed, stats = self.generate_answer(question, context, list(passages))
                parsed["evidencias"] = [
                    {"id": source_id, "cita": passages[source_id]}
                    for source_id in parsed["evidencias"]
                    if isinstance(source_id, str) and source_id in passages
                ]
            else:
                parsed, stats = self.generate_answer(question, context)

        generation_s = (
            time.perf_counter()
            - generation_start
        )

        answer = parsed[
            "respuesta"
        ].strip()

        evidence_declared = parsed[
            "evidencia_suficiente"
        ]

        valid_ids = []
        invalid_ids = []

        # ----------------------------------------------------
        # Validar evidencias
        # ----------------------------------------------------

        for item in parsed["evidencias"]:

            if not isinstance(item, dict):

                invalid_ids.append(
                    str(item)
                )

                continue

            source_id = item.get(
                "id",
                ""
            )

            quote = item.get(
                "cita",
                ""
            )

            if not isinstance(
                source_id,
                str
            ):

                invalid_ids.append(
                    str(source_id)
                )

                continue

            match = re.fullmatch(
                r"S([1-9][0-9]*)",
                source_id
            )

            if not match:

                invalid_ids.append(
                    source_id
                )

                continue

            source_number = int(
                match.group(1)
            )

            if not (
                1 <= source_number <= len(results)
            ):

                invalid_ids.append(
                    source_id
                )

                continue

            if not isinstance(
                quote,
                str
            ):

                invalid_ids.append(
                    source_id
                )

                continue

            source_document = results[
                source_number - 1
            ]

            source_text = source_document.get(
                "text",
                ""
            )

            # Comprobar cita textual
            if not quote_supported(
                quote,
                source_text
            ):

                metadata = (
                    source_document.get("metadata")
                    or {}
                )

                logger.warning(
                    "Cita rechazada | Fuente: %s | "
                    "Cita: %r | Documento: %s | "
                    "Página: %s",
                    source_id,
                    quote,
                    metadata.get("source"),
                    metadata.get("page")
                )

                invalid_ids.append(
                    source_id
                )

                continue

            if source_number not in valid_ids:

                valid_ids.append(
                    source_number
                )

        valid_ids.sort()

        # ----------------------------------------------------
        # Diagnóstico de abstenciones
        # ----------------------------------------------------

        model_abstained = (
            not evidence_declared
            or not answer
            or answer == ABSTENTION
        )

        validation_failed = (
            evidence_declared
            and bool(answer)
            and answer != ABSTENTION
            and not valid_ids
        )

        # ----------------------------------------------------
        # Construcción de respuesta final
        # ----------------------------------------------------

        if (
            model_abstained
            or validation_failed
        ):

            answer = ABSTENTION

            valid_ids = []

        else:

            answer = re.sub(
                r"\s*\[FUENTE\s+\d+\]",
                "",
                answer,
                flags=re.IGNORECASE
            ).strip()

            citations = " ".join(
                f"[FUENTE {i}]"
                for i in valid_ids
            )

            answer = (
                answer + " " + citations
            ).strip()

        # ----------------------------------------------------
        # Detalles de citas
        # ----------------------------------------------------

        citation_details = []

        for source_number in valid_ids:

            metadata = (
                results[source_number - 1]
                .get("metadata")
                or {}
            )

            citation_details.append({
                "source_id": source_number,
                "document": metadata.get("source"),
                "page": metadata.get("page")
            })

        # ----------------------------------------------------
        # Resultado
        # ----------------------------------------------------

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_documents": results,
            "retrieval_mode": self.retrieval_mode,
            "citation_check": {
                "valid_source_ids": valid_ids,
                "invalid_source_ids": invalid_ids,
                "has_valid_source_citation": bool(
                    valid_ids
                ),
                "citation_details": citation_details,
                "citation_count": len(valid_ids),
                "evidence_declared": evidence_declared,
                "evidence_quote_verified": bool(
                    valid_ids
                ),
                "model_abstained": model_abstained,
                "validation_failed": validation_failed
            },
            "timings": {
                "retrieval_s": retrieval_s,
                "generation_s": generation_s,
                "total_s": (
                    time.perf_counter()
                    - start
                ),
                **stats
            }
        }
