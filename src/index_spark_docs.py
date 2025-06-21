import os
import json
from txtai.embeddings import Embeddings

# Este script cria o "Bibliotecário" do nosso sistema RAG. Ele
# converte cada página da documentação em um vetor de embeddings e
# constrói um índice para busca semântica. Esse passo é essencial para
# que, na fase de pergunta, possamos recuperar os trechos de texto mais
# relevantes.

# Configurable paths
DATA_FILE = os.getenv("SCRAPED_FILE", "spark_docs_scrape/spark_guides_dataset_clean.jsonl")
INDEX_PATH = os.getenv("INDEX_PATH", "spark_docs.index")
MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# MODEL_NAME define qual rede pré-treinada usaremos para criar os vetores.
# Modelos como os da família Sentence-Transformers já vêm treinados de
# forma supervisionada para gerar embeddings semânticos eficientes.


def load_documents(path):
    """Load text content from the scraped jsonl file."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            text = obj.get("content")
            if text:
                # txtai espera pares (id, texto). Usamos o indice da linha
                # como id para simplificar.
                docs.append((i, text))
    return docs


def main():
    print(f"Carregando documentos de '{DATA_FILE}'...")
    docs = load_documents(DATA_FILE)
    print(f"{len(docs)} documentos carregados. Criando embeddings...")

    # Aqui geramos um vetor para cada documento usando o modelo pré-treinado.
    embeddings = Embeddings({"path": MODEL_NAME})
    embeddings.index(docs)

    print(f"Salvando índice em '{INDEX_PATH}'...")
    # O índice salvo será usado pelo "Bibliotecário" para recuperar
    # rapidamente os textos mais relevantes durante a inferência.
    embeddings.save(INDEX_PATH)
    print("Indexação concluída.")


if __name__ == "__main__":
    main()
