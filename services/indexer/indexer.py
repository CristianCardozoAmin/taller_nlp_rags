import os
import time
import hashlib
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


def wait_for_solr(timeout: int = 120):
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

    expected_cols = {"seccion", "seccion_principal", "subseccion", "contenido", "contenido_preprocesado"}
    missing = expected_cols - set(df_all.columns)
    if missing:
        raise ValueError(f"Faltan columnas en CSV: {missing}")

    # Limpieza básica
    for c in ["seccion", "seccion_principal", "subseccion", "contenido"]:
        df_all[c] = df_all[c].fillna("").astype(str)

    return df_all


def make_id_doc(sp: str, s: str, ss: str) -> str:
    """ID legible + estable. Ej: 'sp|s|ss::<hash6>' para evitar colisiones."""
    base = f"{sp}|{s}|{ss}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:6]
    return f"{base}::{h}"  # VARCHAR <= 256


def index_solr(df: pd.DataFrame):
    docs: List[dict] = []
    # page = índice 1..N
    for idx, row in df.reset_index(drop=True).iterrows():
        sp = row["seccion_principal"].strip()
        s = row["seccion"].strip()
        ss = row["subseccion"].strip()
        contenido = row["contenido"].strip()

        id_doc = make_id_doc(sp, s, ss)
        page = idx + 1

        doc = {
            "id": f"{id_doc}_{page}",
            "id_doc_s": id_doc,
            "page_i": page,
            "seccion_principal_s": sp,
            "seccion_s": s,
            "subseccion_s": ss,
            "contenido_txt": contenido,  # <- campo de texto para consultas
        }
        docs.append(doc)

    print(f"Enviando {len(docs)} documentos a Solr...")
    payload = {"add": docs}

    r = requests.post(
        f"{SOLR_URL}/update?commit=true",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=600,
    )
    r.raise_for_status()
    print("Indexación en Solr completa.")


def create_or_reset_collection(dim: int) -> Collection:
    if utility.has_collection(MILVUS_COLLECTION):
        print(f"Eliminando colección existente {MILVUS_COLLECTION}...")
        utility.drop_collection(MILVUS_COLLECTION)

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="id_doc", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="seccion_principal", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="seccion", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="subseccion", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="contenido", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="RAG collection (nuevo corpus)")

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
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    test_vec = model.encode(["test"])[0]
    dim = len(test_vec)

    connect_milvus()
    col = create_or_reset_collection(dim)

    batch_size = 128
    num_rows = len(df)
    MAX_CONTENT_LEN = 20000  # puedes declararlo arriba del archivo si quieres

    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        batch = df.iloc[start:end].reset_index(drop=True)
        print(f"Indexando en Milvus filas {start} - {end}...")

        id_docs = []
        pages = []
        sps = []
        ss = []
        sss = []
        contenidos = []

        for i, row in batch.iterrows():
            sp = row["seccion_principal"].strip()
            s = row["seccion"].strip()
            ssub = row["subseccion"].strip()

            cont_raw = row["contenido"]
            if isinstance(cont_raw, str):
                cont_raw = cont_raw.strip()
            else:
                cont_raw = "" if pd.isna(cont_raw) else str(cont_raw)

            # 👇 TRUNCAMOS A 8000 CARACTERES PARA MILVUS
            cont = cont_raw[:MAX_CONTENT_LEN]

            id_doc = make_id_doc(sp, s, ssub)
            page = start + i + 1

            id_docs.append(id_doc)
            pages.append(int(page))
            sps.append(sp)
            ss.append(s)
            sss.append(ssub)
            contenidos.append(cont)

        # Embeddings sobre el texto truncado (suficiente para captar el tema)
        vecs = model.encode(contenidos, batch_size=32, show_progress_bar=False)
        embeds = [v.astype(np.float32).tolist() for v in vecs]

        # El orden de 'entities' debe coincidir EXACTAMENTE con el orden de fields
        entities = [id_docs, pages, sps, ss, sss, contenidos, embeds]
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
