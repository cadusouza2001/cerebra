# index_spark_docs.py

import json
from txtai.embeddings import Embeddings

print("Carregando documentação bruta...")
with open('spark_documentation_raw.json', 'r', encoding='utf-8') as f:
    pages = json.load(f)

# Divide o conteúdo. Uma maneira simples, mas eficaz, é dividir por novas linhas duplas (parágrafos).
chunks = []
for page in pages:
    # Dividindo por parágrafo
    paragraphs = page['content'].split('\n\n')
    for para in paragraphs:
        # Queremos indexar apenas pedaços de texto significativos
        if len(para.strip()) > 100: # Filtra linhas muito curtas/vazias
            chunks.append(para.strip())

print(f"Criados {len(chunks)} pedaços de texto da documentação.")

print("Criando índice de embeddings... (Isso pode levar vários minutos)")
# Este modelo é bom para busca semântica em inglês
embeddings = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2", content=True) # content=True armazena o texto
embeddings.index(chunks)
embeddings.save("spark_docs.index")

print("\nIndexação completa.")
print("Índice de embeddings salvo no diretório 'spark_docs.index'.")