from typing import List, Literal, Optional
import os
import time

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import connections, Collection, utility
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# =========================
# Config
# =========================

SOLR_URL = os.getenv("SOLR_URL", "http://solr:8983/solr/rag")
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_collection")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
openai_client = OpenAI() 

app = FastAPI(title="RAG API unificada", version="0.1.0")


# =========================
# Modelos Pydantic
# =========================

class AskRequest(BaseModel):
    query: str
    backend: Literal["solr", "milvus"] = "solr"
    k: int = 5


class SourceDoc(BaseModel):
    id_doc: str
    page: int
    text_clean: str
    score: float


class AskResponse(BaseModel):
    answer: str
    backend: str
    documents: List[SourceDoc]


# =========================
# Milvus helpers
# =========================

def connect_milvus(retries: int = 10, wait: int = 5):
    """Intenta conectar a Milvus con reintentos, usando host 'milvus' del docker-compose."""
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
            f"La colección {MILVUS_COLLECTION} no existe en Milvus. "
            "Ejecuta primero el servicio 'indexer'."
        )
    col = Collection(MILVUS_COLLECTION)
    col.load()
    return col


# =========================
# Modelo de embeddings (BERT-like)
# =========================

embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global embedding_model
    if embedding_model is None:
        print("Cargando modelo de embeddings (Sentence-BERT)...")
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model


# =========================
# Generador (mismo para ambos backends)
# =========================

def generate_answer(query: str, docs: List[SourceDoc]) -> str:
    """
    Generador unificado usando OpenAI (gpt-5-mini o el modelo que definas).
    """
    # Construimos el contexto a partir de los documentos recuperados
    context = "\n\n".join(
        [
            f"[doc {d.id_doc} pág {d.page} | score={d.score:.4f}]\n{d.text_clean}"
            for d in docs
        ]
    )

    user_prompt = (
        "Responde a la pregunta del usuario usando EXCLUSIVAMENTE la información del contexto.\n"
        "Si no hay información suficiente para responder con certeza, di explícitamente que no se "
        "encuentra la respuesta en las fuentes recuperadas.\n\n"
        f"Pregunta:\n{query}\n\n"
        f"Contexto (fragmentos recuperados):\n{context}\n\n"
        "Instrucciones:\n"
        "- Da una respuesta clara y concisa en español.\n"
        "- Si es útil, menciona explícitamente los documentos/páginas de donde tomaste la información.\n"
        "- No inventes datos que no aparezcan en el contexto.\n"
    )

    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente experto en RAG. Solo usas el contexto proporcionado "
                        "para responder. Si el contexto no contiene la información, lo dices claramente."
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        # Fallback simple si hay algún problema con la llamada a OpenAI
        print(f"Error llamando a OpenAI: {e}")
        answer = (
            "No se pudo generar una respuesta usando el LLM. "
            "Respuesta basada únicamente en concatenar el contexto:\n\n"
            f"Pregunta: {query}\n\n{context}"
        )

    return answer


# =========================
# Búsqueda en Solr (backend léxico)
# =========================

def search_solr(query: str, k: int) -> List[SourceDoc]:
    params = {
        "q": query,
        "rows": k,
        "wt": "json",
        # usamos el campo de texto limpio que indexaremos como text_clean_txt
        "df": "text_clean_txt",
    }
    try:
        resp = requests.get(f"{SOLR_URL}/select", params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando Solr: {e}")
    
    data = resp.json()
    docs = []
    for doc in data.get("response", {}).get("docs", []):

        raw_text  = doc.get("text_clean_txt", "")

        if isinstance(raw_text, list):
            raw_text = " ".join(raw_text)

        text_clean = str(raw_text)

        id_doc = str(doc.get("id_doc_s", ""))
        page = int(doc.get("page_i", 0))
        text_clean = text_clean 
        score = float(doc.get("score", 0.0))
        docs.append(
            SourceDoc(
                id_doc=id_doc,
                page=page,
                text_clean=text_clean,
                score=score,
            )
        )
    return docs


# =========================
# Búsqueda en Milvus (backend vectorial)
# =========================

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
        output_fields=["id_doc", "page", "text_clean"],
    )

    docs: List[SourceDoc] = []
    if not res:
        return docs

    hits = res[0]
    for hit in hits:
        fields = hit.entity
        text_clean = fields.get("text_clean")

        if text_clean is None:
                text_clean = ""
                
        id_doc = str(fields.get("id_doc"))
        page = int(fields.get("page"))
        text_clean = text_clean
        score = float(hit.score)
        docs.append(
            SourceDoc(
                id_doc=id_doc,
                page=page,
                text_clean=text_clean,
                score=score,
            )
        )
    return docs


# =========================
# Endpoint /ask
# =========================

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.k <= 0:
        raise HTTPException(status_code=400, detail="k debe ser > 0")

    if req.backend == "solr":
        docs = search_solr(req.query, req.k)
    elif req.backend == "milvus":
        docs = search_milvus(req.query, req.k)
    else:
        raise HTTPException(
            status_code=400, detail="backend debe ser 'solr' o 'milvus'"
        )

    if not docs:
        raise HTTPException(status_code=404, detail="No se encontraron documentos relevantes")

    answer = generate_answer(req.query, docs)

    return AskResponse(
        answer=answer,
        backend=req.backend,
        documents=docs,
    )
