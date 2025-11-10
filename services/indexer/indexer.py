import os
import time
from typing import List

import numpy as np
import pandas as pd
import requests
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from sentence_transformers import SentenceTransformer

CSV_DIR = os.getenv("CSV_PATH", "/app/data/corpus")
SOLR_URL = os.getenv("SOLR_URL", "http://solr:8983/solr/rag")
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_collection")

# Máximo de caracteres permitido por el campo VARCHAR en Milvus
MAX_TEXT_LEN = 10000

def truncate_text(text, max_len: int = MAX_TEXT_LEN) -> str:
    """Convierte a string y recorta a max_len caracteres."""
    if isinstance(text, str):
        return text[:max_len]
    if pd.isna(text):
        return ""
    return str(text)[:max_len]

def connect_milvus(retries: int = 10, wait: int = 5):
    """Intenta conectar a Milvus con reintentos."""
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


def wait_for_solr(timeout: int = 120):
    """Hace ping a Solr hasta que responda OK o se acabe el timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{SOLR_URL}/admin/ping", timeout=5)
            if r.status_code == 200:
                print("Solr listo.")
                return
        except Exception:
            pass
        print("Esperando a Solr...")
        time.sleep(5)
    raise RuntimeError("Solr no respondió a tiempo.")


def load_csv_files() -> pd.DataFrame:
    files = []
    for name in os.listdir(CSV_DIR):
        if name.lower().endswith(".csv"):
            files.append(os.path.join(CSV_DIR, name))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {CSV_DIR}")

    dfs = []
    for f in files:
        print(f"Cargando {f}...")
        df = pd.read_csv(f, sep="|")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    expected_cols = {"id_doc", "pages", "text", "text_clean", "contenido_preprocesado"}
    missing = expected_cols - set(df_all.columns)
    if missing:
        raise ValueError(f"Faltan columnas en CSV: {missing}")
    
    df_all["text_clean"] = df_all["text_clean"].apply(truncate_text)

    return df_all


def index_solr(df: pd.DataFrame):
    """Indexa documentos en Solr usando campos dinámicos (sin schema.xml)."""
    docs: List[dict] = []

    for _, row in df.iterrows():
        doc = {
            "id": f"{row['id_doc']}_{row['pages']}",  # id único
            "id_doc_s": str(row["id_doc"]),           # string
            "page_i": int(row["pages"]),              # int
            "text_clean_txt": str(row["text_clean"]), # campo de texto
        }
        docs.append(doc)

    print(f"Enviando {len(docs)} documentos a Solr...")
    payload = {"add": docs}

    r = requests.post(
        f"{SOLR_URL}/update?commit=true",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=300,
    )
    r.raise_for_status()
    print("Indexación en Solr completa.")


def create_or_reset_collection(dim: int) -> Collection:
    """Crea (o recrea) la colección en Milvus con el esquema adecuado."""
    if utility.has_collection(MILVUS_COLLECTION):
        print(f"Eliminando colección existente {MILVUS_COLLECTION}...")
        utility.drop_collection(MILVUS_COLLECTION)

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="id_doc", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="text_clean", dtype=DataType.VARCHAR, max_length=65534),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="RAG collection")

    col = Collection(name=MILVUS_COLLECTION, schema=schema)
    print("Colección Milvus creada.")

    col.create_index(
        field_name="embedding",
        index_params={
            "index_type": "AUTOINDEX",
            "metric_type": "IP",
            "params": {},
        },
    )
    print("Índice de Milvus creado.")
    return col


def index_milvus(df: pd.DataFrame):
    """Genera embeddings BERT y los indexa en Milvus."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    test_vec = model.encode(["test"])[0]
    dim = len(test_vec)

    connect_milvus()
    col = create_or_reset_collection(dim)

    batch_size = 128
    num_rows = len(df)

    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        batch = df.iloc[start:end]
        print(f"Indexando en Milvus filas {start} - {end}...")

        id_docs = [str(x) for x in batch["id_doc"].tolist()]
        pages = [int(x) for x in batch["pages"].tolist()]
        texts = [str(x) for x in batch["text_clean"].tolist()]

        embeds = model.encode(texts, batch_size=32, show_progress_bar=False)
        embeds = [v.astype(np.float32).tolist() for v in embeds]

        entities = [id_docs, pages, texts, embeds]
        col.insert(entities)

    col.flush()
    col.load()
    print("Indexación en Milvus completa.")


def main():
    print("Esperando servicios...")
    wait_for_solr()
    connect_milvus()

    df = load_csv_files()

    index_solr(df)
    index_milvus(df)

    print("Indexación completa. Saliendo.")


if __name__ == "__main__":
    main()
