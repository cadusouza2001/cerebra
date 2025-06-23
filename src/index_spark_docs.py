import os
import json
from txtai.embeddings import Embeddings

# Este módulo lida com a parte de "recuperação" (Retrieval) do
# RAG. Geramos embeddings semânticos para cada página da
# documentação, permitindo depois buscar trechos relevantes para
# uma pergunta.

# Este script cria o "Bibliotecário" do nosso sistema RAG. Ele
# converte cada página da documentação em um vetor de embeddings e
# constrói um índice para busca semântica. Esse passo é essencial para
# que, na fase de pergunta, possamos recuperar os trechos de texto mais
# relevantes.
# Os embeddings funcionam como a representação vetorial de cada
# documento, conceito que estudamos em NLP para permitir medidas de
# similaridade entre textos.

DATA_FILE = os.getenv("SCRAPED_FILE", "spark_docs_scrape/spark_guides_dataset_clean.jsonl")
INDEX_PATH = os.getenv("INDEX_PATH", "spark_docs.index")
# MODEL_NAME define qual rede pré-treinada usaremos para criar os vetores.
# Modelos como os da família Sentence-Transformers já vêm treinados de
# forma supervisionada para gerar embeddings semânticos eficientes.
MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDINGS_OUTPUT = os.getenv("EMBEDDINGS_OUTPUT", "doc_embeddings.npz")


def load_documents(path):
    """Carrega o conteúdo de texto de um jsonl"""
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
    # O processo é similar à etapa de vetorização vista nas aulas de NLP.
    # Ativamos a opcao 'content' para que o texto original seja salvo junto com
    # os vetores. Assim, buscas posteriores retornam tambem os trechos de texto.
    embeddings = Embeddings({"path": MODEL_NAME, "content": True})
    embeddings.index(docs)

    # Salvamos também os vetores para geração de gráficos (PCA/t-SNE)
    import numpy as np
    vectors = embeddings.transform([t for _, t in docs])
    np.savez_compressed(EMBEDDINGS_OUTPUT, vectors=vectors)

    print(f"Salvando índice em '{INDEX_PATH}'...")
    # O índice salvo será usado pelo "Bibliotecário" para recuperar
    # rapidamente os textos mais relevantes durante a inferência.
    embeddings.save(INDEX_PATH)
    print("Indexação concluída.")


if __name__ == "__main__":
    main()
