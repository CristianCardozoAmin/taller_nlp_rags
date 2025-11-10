import os
import time
import json
from typing import List, Dict
from openai import OpenAI
import numpy as np
import requests
import matplotlib.pyplot as plt

API_URL = os.getenv("API_URL", "http://api:8000")
REPORT_DIR = os.getenv("REPORT_DIR", "/app/reports")
EVAL_FILE = os.getenv("EVAL_FILE", "/app/data/eval_queries.json")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
llm_client = OpenAI()   


os.makedirs(REPORT_DIR, exist_ok=True)


def load_eval_data() -> List[Dict]:
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    inter = len(retrieved_set & relevant_set)
    return inter / len(relevant_set)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    rel_set = set(relevant_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids, start=1):
        rel = 1.0 if rid in rel_set else 0.0
        dcg += (2**rel - 1) / np.log2(i + 1)
    ideal_len = min(len(retrieved_ids), len(rel_set))
    idcg = 0.0
    for i in range(1, ideal_len + 1):
        idcg += (2**1 - 1) / np.log2(i + 1)
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---------- ROUGE-L simple ----------

def lcs(a: List[str], b: List[str]) -> int:
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]


def rouge_l(hyp: str, ref: str) -> float:
    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs_len = lcs(hyp_tokens, ref_tokens)
    prec = lcs_len / len(hyp_tokens)
    rec = lcs_len / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    beta = 1.0
    return (1 + beta**2) * prec * rec / (rec + prec)


# ---------- LLM-as-a-judge (placeholder) ----------

def llm_as_judge_score(query: str, answer: str, ideal_answer: str) -> float:
    """
    Usa un LLM de OpenAI como juez para evaluar la calidad de la respuesta.
    Devuelve un score entre 0.0 y 1.0.

    El prompt le pide al modelo que DEVUELVA SOLO UN NÚMERO entre 0 y 1.
    """
    if not ideal_answer:
        return 0.0

    judge_prompt = (
        "Eres un evaluador estricto de respuestas de un sistema de recuperación aumentada (RAG).\n"
        "Tu tarea es comparar la respuesta del sistema con una respuesta ideal y producir un "
        "score numérico entre 0 y 1.\n\n"
        "Instrucciones:\n"
        "- 1.0 significa que la respuesta del sistema es prácticamente perfecta, está bien "
        "  fundamentada y coincide con la respuesta ideal.\n"
        "- 0.0 significa que la respuesta es completamente incorrecta, irrelevante o inventada.\n"
        "- Valores intermedios reflejan grados de calidad.\n"
        "- IMPORTANTE: Tu salida DEBE ser exclusivamente un número entre 0 y 1 "
        "  (por ejemplo: 0.0, 0.25, 0.73, 1.0). Sin texto adicional.\n\n"
        f"Pregunta del usuario:\n{query}\n\n"
        f"Respuesta del sistema:\n{answer}\n\n"
        f"Respuesta ideal (de referencia):\n{ideal_answer}\n\n"
        "Devuelve solo el número del score:"
    )

    try:
        completion = llm_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un juez muy estricto. Siempre devuelves solo un número entre 0 y 1 "
                        "como texto plano."
                    ),
                },
                {
                    "role": "user",
                    "content": judge_prompt,
                },
            ],
        )
        raw = completion.choices[0].message.content.strip()
        # Intentamos parsear el número que devuelva el modelo
        score = None
        # Primero intenta parsear todo
        try:
            score = float(raw.replace(",", "."))
        except ValueError:
            # Si el modelo devolvió algo más, buscamos el primer token que parezca número
            for tok in raw.split():
                try:
                    score = float(tok.replace(",", "."))
                    break
                except ValueError:
                    continue

        if score is None:
            return 0.0

        # Clamp a [0,1]
        score = max(0.0, min(1.0, score))
        return float(score)
    except Exception as e:
        print(f"Error en LLM-as-a-judge: {e}")
        return 0.0


# ---------- Evaluación de backend ----------

def evaluate_backend(backend: str, eval_data: List[Dict]) -> Dict[str, float]:
    latencies = []
    recalls = []
    mrrs = []
    ndcgs = []
    rouges = []
    judges = []

    for item in eval_data:
        query = item["query"]
        relevant_ids = item.get("relevant_docs", [])
        ideal_answer = item.get("ideal_answer", "")

        t0 = time.time()
        resp = requests.post(
            f"{API_URL}/ask",
            json={"query": query, "backend": backend, "k": 10},
            timeout=120,
        )
        dt = time.time() - t0
        latencies.append(dt)

        if resp.status_code != 200:
            print(
                f"Error {resp.status_code} para backend {backend} "
                f"query '{query}': {resp.text}"
            )
            continue

        data = resp.json()
        docs = data.get("documents", [])
        retrieved_ids = [f"{d['id_doc']}_{d['page']}" for d in docs]
        answer = data.get("answer", "")

        recalls.append(recall_at_k(retrieved_ids, relevant_ids))
        mrrs.append(mrr(retrieved_ids, relevant_ids))
        ndcgs.append(ndcg(retrieved_ids, relevant_ids))

        if ideal_answer:
            rouges.append(rouge_l(answer, ideal_answer))
            judges.append(llm_as_judge_score(query, answer, ideal_answer))

    def avg(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return {
        "latency_avg": avg(latencies),
        "recall@10": avg(recalls),
        "mrr": avg(mrrs),
        "ndcg": avg(ndcgs),
        "rouge_l": avg(rouges),
        "llm_as_judge": avg(judges),
    }


def save_results(results: Dict[str, Dict[str, float]]):
    out_path = os.path.join(REPORT_DIR, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en {out_path}")

    backends = list(results.keys())
    metrics = ["latency_avg", "recall@10", "mrr", "ndcg", "rouge_l", "llm_as_judge"]

    for metric in metrics:
        values = [results[b][metric] for b in backends]
        plt.figure()
        plt.bar(backends, values)
        plt.title(metric)
        plt.ylabel(metric)
        plt.tight_layout()
        fig_path = os.path.join(REPORT_DIR, f"{metric}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Gráfico {metric} guardado en {fig_path}")


def main():
    eval_data = load_eval_data()
    results = {}
    for backend in ["solr", "milvus"]:
        print(f"Evaluando backend {backend}...")
        results[backend] = evaluate_backend(backend, eval_data)
    save_results(results)
    print("Evaluación completa.")


if __name__ == "__main__":
    main()