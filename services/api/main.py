from typing import List, Literal, Optional
import os
import time

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import connections, Collection, utility

from sentence_transformers import SentenceTransformer
from openai import OpenAI  # (si ya integraste el LLM)

SOLR_URL = os.getenv("SOLR_URL", "http://solr:8983/solr/rag")
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_collection")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
openai_client = OpenAI()

app = FastAPI(title="RAG API unificada", version="0.2.0")

class AskRequest(BaseModel):
    query: str
    backend: Literal["solr", "milvus"] = "solr"
    k: int = 5

class SourceDoc(BaseModel):
    id_doc: str
    page: int
    contenido: str   # CAMBIO
    score: float

class AskResponse(BaseModel):
    answer: str
    backend: str
    documents: List[SourceDoc]

def connect_milvus(retries: int = 10, wait: int = 5):
    for i in range(retries):
        try:
            print(f"Conectando a Milvus (intento {i+1}/{retries})...")
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
            print("Conexión a Milvus exitosa.")
            return
        except Exception as e:
            print(f"Error conectando a Milvus: {e}")
            if i == retries - 1:
                raise
            time.sleep(wait)

def get_milvus_collection() -> Collection:
    if not connections.has_connection("default"):
        connect_milvus()
    if not utility.has_collection(MILVUS_COLLECTION):
        raise RuntimeError(
            f"La colección {MILVUS_COLLECTION} no existe en Milvus. Ejecuta 'indexer' primero."
        )
    col = Collection(MILVUS_COLLECTION)
    col.load()
    return col

embedding_model: Optional[SentenceTransformer] = None
def get_embedding_model() -> SentenceTransformer:
    global embedding_model
    if embedding_model is None:
        print("Cargando modelo de embeddings (Sentence-BERT)...")
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

def generate_answer(query: str, docs: List[SourceDoc]) -> str:
    # Usa el LLM (mismo para ambos backends)
    context = "\n\n".join(
        [f"[{d.id_doc} pág {d.page} | score={d.score:.4f}]\n{d.contenido}" for d in docs]  # CAMBIO
    )
    user_prompt = (
        "Responde a la pregunta del usuario usando EXCLUSIVAMENTE la información del contexto.\n"
        "Si no hay suficiente información, dilo explícitamente.\n\n"
        f"Pregunta:\n{query}\n\n"
        f"Contexto:\n{context}\n"
    )
    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Eres un asistente experto en RAG."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error llamando a OpenAI: {e}")
        return f"(Fallback) No fue posible usar el LLM. Contexto:\n{context}"

def search_solr(query: str, k: int) -> List[SourceDoc]:
    params = {
        "q": query,
        "rows": k,
        "wt": "json",
        "df": "contenido_txt",
    }
    try:
        resp = requests.get(f"{SOLR_URL}/select", params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando Solr: {e}")

    data = resp.json()
    out: List[SourceDoc] = []

    for doc in data.get("response", {}).get("docs", []):
        raw_contenido = doc.get("contenido_txt", "")

        if isinstance(raw_contenido, list):
            contenido = " ".join(str(x) for x in raw_contenido)
        else:
            contenido = str(raw_contenido)

        out.append(
            SourceDoc(
                id_doc=str(doc.get("id_doc_s", "")),
                page=int(doc.get("page_i", 0)),
                contenido=contenido,  # ahora SIEMPRE es string
                score=float(doc.get("score", 0.0)),
            )
        )

    return out


def search_milvus(query: str, k: int) -> List[SourceDoc]:
    model = get_embedding_model()
    vec = model.encode([query])[0].astype(np.float32).tolist()

    col = get_milvus_collection()

    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10},
    }

    res = col.search(
        data=[vec],
        anns_field="embedding",
        param=search_params,
        limit=k,
        output_fields=["id_doc", "page", "contenido"],  # asegurarse de que estos campos existen
    )

    docs: List[SourceDoc] = []
    if not res:
        return docs

    hits = res[0]
    for hit in hits:
        entity = hit.entity  # este NO es un dict, es un Entity

        # get() solo recibe el nombre del campo
        id_doc_val = entity.get("id_doc")
        page_val = entity.get("page")
        contenido_val = entity.get("contenido")

        id_doc = str(id_doc_val) if id_doc_val is not None else ""
        page = int(page_val) if page_val is not None else 0
        contenido = contenido_val if contenido_val is not None else ""

        docs.append(
            SourceDoc(
                id_doc=id_doc,
                page=page,
                contenido=contenido,
                score=float(hit.score),
            )
        )

    return docs


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k debe ser > 0")
    if req.backend == "solr":
        docs = search_solr(req.query, req.k)
    elif req.backend == "milvus":
        docs = search_milvus(req.query, req.k)
    else:
        raise HTTPException(status_code=400, detail="backend debe ser 'solr' o 'milvus'")
    if not docs:
        raise HTTPException(status_code=404, detail="No se encontraron documentos relevantes")
    answer = generate_answer(req.query, docs)
    return AskResponse(answer=answer, backend=req.backend, documents=docs)
