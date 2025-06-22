from __future__ import annotations

import json
import os
import time
from multiprocessing import Lock, Process, Queue
from pathlib import Path
from typing import Iterable, Tuple
import random

import google.generativeai as genai

"""Gera pares de Pergunta e Resposta de forma assíncrona usando o Gemini.

Aceita múltiplas chaves de API em paralelo e registra quais
requisições foram bem-sucedidas ou falharam. Se a execução for interrompida,
basta rodar novamente para continuar do ponto em que parou.
"""


# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

INPUT_DATA_FILE = "spark_docs_scrape/spark_guides_dataset_clean.jsonl"
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "qa_dataset/spark_qa_generative_dataset.jsonl")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))
CHUNKS_PER_REQUEST = int(os.getenv("CHUNKS_PER_REQUEST", "30"))

SUCCESS_LOG = os.getenv("SUCCESS_LOG", "qa_dataset/processed_chunks.log")
FAILED_LOG = os.getenv("FAILED_LOG", "qa_dataset/failed_chunks.log")

# A API Gemini permite 30 requisições por minuto por chave
RATE_LIMIT = 30  # requisições por minuto (por chave)

# Recupera múltiplas chaves separadas por vírgula
API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
if not API_KEYS:
    single_key = os.getenv("GEMINI_API_KEY")
    if single_key:
        API_KEYS = [single_key]
    else:
        raise RuntimeError(
            "Nenhuma chave encontrada. Defina GEMINI_API_KEYS ou GEMINI_API_KEY."
        )

INTERVAL = 60.0 / RATE_LIMIT

# Termos técnicos que serão usados para validação das perguntas
TECH_TERMS = {
    "dataframe",
    "dataset",
    "rdd",
    "cluster",
    "executor",
    "partition",
    "spark",
}

QUESTION_STYLES = [
    "explain",
    "when to use",
    "why",
    "how does it work",
]


def random_style() -> str:
    return random.choice(QUESTION_STYLES)


def classify_chunk(text: str) -> str:
    """Classifica a dificuldade do chunk por heurística."""
    tokens = text.split()
    term_count = sum(1 for term in TECH_TERMS if term in text.lower())
    if len(tokens) < 50 and term_count < 3:
        return "basic"
    if len(tokens) < 120:
        return "intermediate"
    return "advanced"


def quality_score(question: str, answer: str) -> float:
    """Atribui um score simples de qualidade baseado no tamanho e termos."""
    length_score = min(1.0, (len(question.split()) + len(answer.split())) / 150)
    tech_score = (
        sum(term in (question + " " + answer).lower() for term in TECH_TERMS)
        / len(TECH_TERMS)
    )
    return round((length_score + tech_score) / 2, 3)


def validate_qa_pair(pair: dict) -> bool:
    """Verifica se o par segue os critérios mínimos de qualidade."""
    if not pair or "question" not in pair or "answer" not in pair:
        return False
    q_tokens = pair["question"].split()
    a_tokens = pair["answer"].split()
    if len(q_tokens) < 10 or len(a_tokens) < 10:
        return False
    def uniq_ratio(tokens: list[str]) -> float:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    if uniq_ratio(q_tokens) < 0.5 or uniq_ratio(a_tokens) < 0.5:
        return False
    text = (pair["question"] + " " + pair["answer"]).lower()
    if sum(term in text for term in TECH_TERMS) < 2:
        return False
    return True


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def create_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide um texto longo em pedaços menores e sobrepostos."""
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def call_llm_for_qa(model: genai.GenerativeModel, text_chunk: str) -> dict | None:
    """Envia o trecho de texto ao modelo e retorna o par QA gerado."""

    prompt = f"""
    Your task is to act as a helpful expert assistant who creates high-quality question-and-answer pairs from a given technical text.
    Generate exactly one question and a complete, helpful answer.

    CRITICAL INSTRUCTIONS:
    1. The 'answer' should be a full, natural-sounding sentence or paragraph. It should be helpful and comprehensive. For example, if the question is "What is the command?", the answer should be "The command to use is `...`", not just "`...`".
    2. The answer MUST be factually grounded in the provided "Text". Do not make up information.
    3. Your response MUST BE ONLY a single, valid JSON object with the keys "question" and "answer".

    EXAMPLE:
    Text: "The --master option specifies the master URL for a distributed cluster, or local to run locally with one thread."
    Your JSON Response:
    {{
      "question": "What is the purpose of the --master option in Spark?",
      "answer": "The --master option is used to specify the master URL for a distributed cluster. You can also use 'local' to run it on a single thread locally."
    }}

    Now, process the following text:
    ---
    Text: "{text_chunk}"
    ---
    Your JSON Response:
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.35),
            request_options={"timeout": 100},
        )

        if not response.parts:
            print("  └─ [AVISO] Resposta da API bloqueada.")
            return None

        json_text = (
            response.text.strip().replace("```json", "").replace("```", "")
        )
        qa_pair = json.loads(json_text)

        if "question" in qa_pair and "answer" in qa_pair:
            return qa_pair
        print(
            "  └─ [AVISO] JSON retornado não continha as chaves esperadas. Ignorado."
        )
        return None
    except Exception as exc: 
        print(f"  └─ [ERRO] Falha na chamada da API: {exc}")
        return None


def call_llm_for_qa_batch(
    model: genai.GenerativeModel, text_chunks: list[str]
) -> list[dict | None]:
    """Envia vários trechos e retorna uma lista de pares QA."""

    if not text_chunks:
        return []

    difficulties = [classify_chunk(c) for c in text_chunks]
    styles = [random_style() for _ in text_chunks]
    texts = "\n\n".join(
        [
            f"Text {i+1} (difficulty: {d}, style: {s}): \"{c}\""
            for i, (c, d, s) in enumerate(zip(text_chunks, difficulties, styles))
        ]
    )

    prompt = f"""
    You are an expert in Apache Spark. For each provided text chunk, generate exactly one question and a complete, pedagogically helpful answer.
    Each text comes with a suggested style (after the word 'style:'), which you must use when formulating the question.
    Use the given difficulty label to adjust complexity:
    - basic: formulate a straightforward question.
    - intermediate: formulate a slightly deeper or "why/how" style question, possibly multiple choice.
    - advanced: formulate an in-depth or comparative question that encourages detailed reasoning.
    Enrich the answer with short analogies or examples when relevant and keep it grounded in the text.

    CRITICAL INSTRUCTIONS:
    1. Each answer must be grounded in its respective text.
    2. Return a JSON array in the same order of the input texts, where each element has the keys 'question' and 'answer'.

    Example:
    Text 1: "Example text"
    Text 2: "Another text"
    Your JSON Response:
    [
      {{"question": "Q1", "answer": "A1"}},
      {{"question": "Q2", "answer": "A2"}}
    ]

    Now, process the following texts:
    ---
    {texts}
    ---
    Your JSON Response:
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.35),
            request_options={"timeout": 100},
        )

        if not response.parts:
            print("  └─ [AVISO] Resposta da API bloqueada.")
            return [None for _ in text_chunks]

        json_text = (
            response.text.strip().replace("```json", "").replace("```", "")
        )

        data = json.loads(json_text)
        if isinstance(data, list):
            results: list[dict | None] = []
            for item in data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    results.append(item)
                else:
                    results.append(None)
            # pad if fewer results were returned
            if len(results) < len(text_chunks):
                results.extend([None] * (len(text_chunks) - len(results)))
            return results

        print("  └─ [AVISO] Formato inesperado de resposta. Ignorado.")
        return [None for _ in text_chunks]
    except Exception as exc:
        print(f"  └─ [ERRO] Falha na chamada da API: {exc}")
        return [None for _ in text_chunks]


def load_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Processamento em paralelo
# ---------------------------------------------------------------------------

def worker(api_key: str, queue: Queue, lock: Lock) -> None:
    """Processa chunks usando a chave especificada."""

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-27b-it")

    last_request = 0.0

    while True:
        batch: list[Tuple[int, str]] = []
        # obtém pelo menos um item ou detecta fim
        item: Tuple[int, str] | None = queue.get()
        if item is None:
            break
        batch.append(item)

        # pega mais itens até atingir o tamanho desejado, sem bloquear se não houver
        while len(batch) < CHUNKS_PER_REQUEST and not queue.empty():
            nxt = queue.get()
            if nxt is None:
                queue.put(None)
                break
            batch.append(nxt)

        wait = INTERVAL - (time.time() - last_request)
        if wait > 0:
            time.sleep(wait)

        qa_pairs = call_llm_for_qa_batch(model, [c for _, c in batch])
        last_request = time.time()

        with lock:
            for (idx, chunk), qa_pair in zip(batch, qa_pairs):
                if qa_pair and validate_qa_pair(qa_pair):
                    qa_pair["source_chunk"] = chunk
                    qa_pair["chunk_index"] = idx
                    qa_pair["quality_score"] = quality_score(
                        qa_pair["question"], qa_pair["answer"]
                    )
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                    with open(SUCCESS_LOG, "a", encoding="utf-8") as f:
                        f.write(f"{idx}\n")
                else:
                    with open(FAILED_LOG, "a", encoding="utf-8") as f:
                        f.write(f"{idx}\n")


def main() -> None:
    for path in (OUTPUT_FILE, SUCCESS_LOG, FAILED_LOG):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_FILE).touch(exist_ok=True)

    print(f"Carregando documentação de '{INPUT_DATA_FILE}'...")
    scraped_pages = list(load_jsonl(INPUT_DATA_FILE))

    print("Dividindo em pedaços (chunks)...")
    all_chunks: list[str] = []
    for page in scraped_pages:
        if page.get("content"):
            all_chunks.extend(create_chunks(page["content"]))

    print(f"Total de {len(all_chunks)} chunks gerados.")

    processed = set()
    if Path(SUCCESS_LOG).exists():
        processed = {int(l.strip()) for l in open(SUCCESS_LOG, "r", encoding="utf-8") if l.strip()}

    failed_previous: list[int] = []
    if Path(FAILED_LOG).exists():
        failed_previous = [int(l.strip()) for l in open(FAILED_LOG, "r", encoding="utf-8") if l.strip()]

    queue: Queue[Tuple[int, str]] = Queue()

    # Reprocessa primeiro os que falharam anteriormente
    for idx in failed_previous:
        if idx not in processed and idx < len(all_chunks):
            queue.put((idx, all_chunks[idx]))

    # Processa novos chunks ainda não executados
    for idx, chunk in enumerate(all_chunks):
        if idx not in processed and idx not in failed_previous:
            queue.put((idx, chunk))

    # Sentinelas para encerrar os processos
    for _ in API_KEYS:
        queue.put(None)

    # Limpa o arquivo de falhas para a nova execução
    Path(FAILED_LOG).write_text("")

    lock = Lock()
    processes = [Process(target=worker, args=(key, queue, lock)) for key in API_KEYS]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("\nProcesso concluído.")
    total_success = len({int(l.strip()) for l in open(SUCCESS_LOG, 'r', encoding='utf-8') if l.strip()})
    print(f"Total de pares gerados com sucesso: {total_success}")
    remaining_failures = len([l for l in open(FAILED_LOG, 'r', encoding='utf-8') if l.strip()])
    if remaining_failures:
        print(f"Falharam {remaining_failures} chunks. Consulte '{FAILED_LOG}' para reprocessar.")


if __name__ == "__main__":  
    main()
