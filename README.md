# 1. Cerebra 🧠

## 1.1. Visão Geral do Projeto

O **Cerebra** é um sistema simples de Perguntas e Respostas (Q&A) construído em Python. Ele transforma uma coleção de páginas da web em um especialista capaz de responder a questões sobre aquele conteúdo. Todo o modelo é treinado do zero utilizando PyTorch e dados gerados automaticamente com a API Gemini.

### 1.1.1. Tecnologias e bibliotecas

- Python 3.9+
- `aiohttp` e `BeautifulSoup4` para raspagem
- `txtai` para busca por similaridade (Retriever)
- `torch` para treinar a rede neural
- `google-generativeai` para criar pares de pergunta e resposta

## 1.2. Estrutura do Projeto

```
.
├── qa_dataset/                    # Arquivos de treinamento
│   └── spark_qa_generative_dataset.jsonl
├── spark_docs_scrape/             # Dados raspados da documentação
│   ├── spark_guides_dataset_clean.jsonl
│   ├── visited_urls_clean.log
│   └── visited_urls_guides.log
└── src/
    ├── demo.py                   # Exemplo rápido de uso
    ├── evaluate_model.py         # Avaliação qualitativa
    ├── generate_qa_dataset.py    # Cria o dataset com a API Gemini
    ├── index_spark_docs.py       # Gera o índice semântico
    ├── qa_model.py               # Modelo seq2seq em PyTorch
    ├── run_qa_system.py          # Perguntas para o modelo treinado
    ├── scrape_fast_resumable.py  # Raspagem assíncrona da documentação
    └── train.py                  # Treino do modelo do zero
```

## 1.3. Execução no Google Colab

1. **Preparação do ambiente**

   - Abra um novo notebook no Google Colab e ative o modo **GPU** em `Runtime > Change runtime type`.
   - Clone este repositório:
     ```python
     !git clone https://github.com/cadusouza2001/cerebra.git
     %cd cerebra
     ```
   - Instale as dependências:
     ```python
     !pip install aiohttp beautifulsoup4 txtai torch google-generativeai
     ```
   - (Opcional) Monte o Google Drive para salvar outputs:

     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

2. **Variáveis de ambiente**

   - Use o recurso **Secrets** do Colab (ícone de chave na lateral):
   - Adicione a variável `GOOGLE_API_KEY` via interface.
   - No código Colab, recupere com:

     ```python
     from google.colab import userdata
     import os

     gemini_key = userdata.get("GOOGLE_API_KEY")
     os.environ["GEMINI_API_KEY"] = gemini_key
     ```

   - Isso garante que a chave é mantida privada e não inserida diretamente no código.
   - Outras variáveis podem ser configuradas da mesma forma, por exemplo:
     ```python
     os.environ["DATASET_FILE"] = "qa_dataset/spark_qa_generative_dataset.jsonl"
     os.environ["OUTPUT_MODEL_DIR"] = "spark_expert_model"
     ```

3. **Passo a passo**
   Execute os scripts na ordem abaixo, cada um em uma célula do Colab:
   1. Raspagem da documentação
      ```python
      !python src/scrape_fast_resumable.py
      ```
   2. Criação do índice semântico
      ```python
      !python src/index_spark_docs.py
      ```
   3. Geração automática do conjunto de dados
      ```python
      !python src/generate_qa_dataset.py
      ```
   4. Treinamento do modelo (usa GPU se disponível)
      ```python
      !python src/train.py
      ```
   5. Avaliação rápida
      ```python
      !python src/evaluate_model.py
      ```
   6. Perguntas interativas
      ```python
      !python src/run_qa_system.py
      ```

Os arquivos resultantes são gravados dentro do próprio diretório do projeto. O índice fica em `spark_docs.index` e o modelo treinado em `spark_expert_model/model.pt`.

## 1.4. Licença e Créditos

Este projeto está licenciado sob a licença MIT e foi desenvolvido originalmente por Carlos Souza e colaboradores para fins educacionais. Sinta-se livre para estudar, modificar e compartilhar. Consulte o arquivo `LICENSE` para mais detalhes.
