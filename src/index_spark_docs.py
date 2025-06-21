import os
import json
from txtai.embeddings import Embeddings

# Configurable paths
DATA_FILE = os.getenv("SCRAPED_FILE", "spark_docs_scrape/spark_guides_dataset_clean.jsonl")
INDEX_PATH = os.getenv("INDEX_PATH", "spark_docs.index")
MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def load_documents(path):
    """Load text content from the scraped jsonl file."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            text = obj.get("content")
            if text:
                docs.append((i, text))
    return docs


def main():
    print(f"Carregando documentos de '{DATA_FILE}'...")
    docs = load_documents(DATA_FILE)
    print(f"{len(docs)} documentos carregados. Criando embeddings...")

    embeddings = Embeddings({"path": MODEL_NAME})
    embeddings.index(docs)

    print(f"Salvando índice em '{INDEX_PATH}'...")
    embeddings.save(INDEX_PATH)
    print("Indexação concluída.")


if __name__ == "__main__":
    main()
