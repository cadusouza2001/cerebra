# Cerebra 🧠

[](https://www.google.com/search?q=LICENSE)
[](https://www.python.org/downloads/)
[](https://huggingface.co/transformers)

**Cerebra** é um framework poderoso e flexível para construir sistemas de Perguntas e Respostas (Q&A) especialistas, capazes de dominar qualquer documentação ou base de conhecimento.

Em vez de simplesmente buscar por palavras-chave, o Cerebra usa uma arquitetura moderna de IA para ler, entender e raciocinar sobre o conteúdo, fornecendo respostas precisas e contextuais. Este projeto foi projetado para ser adaptável: aponte-o para uma nova documentação, e ele aprenderá a se tornar um especialista nesse novo domínio.

## ✨ Conceitos Principais

O Cerebra é construído sobre uma arquitetura de **Geração Aumentada por Recuperação (Retrieval-Augmented Generation - RAG)**. Isso funciona de forma análoga a um pesquisador humano especialista:

1.  **O Bibliotecário (Retriever):** Primeiro, um componente de busca semântica ultrarrápido (o "Bibliotecário") varre toda a base de conhecimento para encontrar os trechos de texto mais relevantes para a pergunta do usuário.
2.  **O Especialista (Reader):** Em seguida, esses trechos relevantes são entregues a um modelo de linguagem avançado e ajustado (o "Especialista"), que lê o contexto focado e formula uma resposta precisa e bem fundamentada.

Essa abordagem de dois estágios torna o sistema escalável, preciso e muito menos propenso a "alucinações" ou respostas incorretas.

## 🚀 Funcionalidades

  * **Adaptável a Qualquer Domínio:** Aponte para qualquer site ou conjunto de documentos para criar um novo especialista.
  * **Busca Semântica:** Entende o *significado* por trás da sua pergunta, não apenas as palavras-chave.
  * **Geração de Dados Automatizada:** Usa a API Gemini do Google para gerar automaticamente um conjunto de dados de treinamento de alta qualidade a partir de qualquer texto.
  * **Modelo Especialista Ajustado:** Treina um modelo de linguagem para se tornar um especialista no jargão e nos conceitos do seu domínio específico.
  * **Arquitetura Modular:** Cada etapa (raspagem, indexação, treinamento, inferência) é separada em scripts claros e reutilizáveis.

## 🛠️ Tech Stack

  * **Backend:** Python 3.9+
  * **IA & NLP:**
      * Hugging Face `transformers` & `datasets`
      * `txtai` para busca de embeddings
      * Google `generativeai` (para a API Gemini Pro)
      * `torch` (PyTorch)
  * **Raspagem de Dados:** `requests` & `BeautifulSoup4`

## 📂 Estrutura do Projeto

```
.
├── scrape_spark_docs.py      # 1. Coleta o conteúdo do site
├── index_spark_docs.py       # 2. Processa e cria o índice de busca
├── generate_qa_dataset.py    # 3. Gera o dataset de P&R com Gemini
├── train_spark_expert.py     # 4. Treina o modelo especialista
└── run_spark_qa_system.py    # 5. Executa o sistema final para responder perguntas
```

## 🏁 Guia de Início Rápido

Siga estes passos para ter sua própria instância do Cerebra funcionando.

### 1\. Pré-requisitos

  * Python 3.9 ou superior
  * Uma chave de API do Google Gemini (obtenha em [Google AI Studio](https://aistudio.google.com/app/apikey))

### 2\. Instalação

Primeiro, clone o repositório e navegue até o diretório:

```bash
git clone https://github.com/SEU-USUARIO/cerebra.git
cd cerebra
```

Crie e ative um ambiente virtual (recomendado):

```bash
python -m venv venv
source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
```

Instale todas as dependências necessárias:

```bash
pip install requests beautifulsoup4 "txtai[pipeline]" google-generativeai torch transformers datasets
```

### 3\. Configuração

Defina a variável de ambiente `GEMINI_API_KEY` com sua chave de API do Gemini.
O script `generate_qa_dataset.py` irá lê-la automaticamente.

## 📈 Ordem de Execução

Execute os scripts na seguinte ordem para construir e iniciar o sistema.

1.  **Coletar a Documentação:**

      * *Opcional: Edite `scrape_spark_docs.py` para apontar para a URL da documentação que você deseja.*

    <!-- end list -->

    ```bash
    python scrape_spark_docs.py
    ```

2.  **Indexar o Conteúdo:**

    ```bash
    python index_spark_docs.py
    ```

3.  **Gerar o Conjunto de Dados de Treinamento:**

      * *Este passo usa a API do Gemini e pode incorrer em custos.*

    <!-- end list -->

    ```bash
    python generate_qa_dataset.py
    ```

4.  **Treinar o Modelo Especialista:**

      * *Este passo é computacionalmente intensivo e é **altamente recomendado** executá-lo em um ambiente com GPU (por exemplo, Google Colab).*

    <!-- end list -->

    ```bash
    python train_spark_expert.py
    ```

5.  **Executar o Sistema de P\&R:**

      * *Depois que tudo estiver construído, execute este script para começar a fazer perguntas\!*

    <!-- end list -->

    ```bash
    python run_spark_qa_system.py
    ```

## 🗺️ Roadmap Futuro

O Cerebra é um framework com enorme potencial. As próximas etapas poderiam incluir:

  * [ ] **Interface Web:** Construir uma interface de usuário amigável com Streamlit ou Flask.
  * [ ] **Suporte a Mais Formatos:** Adicionar a capacidade de ingerir documentos de PDFs, arquivos Markdown e outros formatos.
  * [ ] **Containerização:** Empacotar a aplicação com Docker para facilitar a implantação.
  * [ ] **Scripts de Avaliação:** Adicionar um script para medir a precisão e a eficácia do modelo treinado.

## 🤝 Contribuições

Contribuições são bem-vindas! Se você tiver ideias para novas funcionalidades ou melhorias, sinta-se à vontade para abrir uma *Issue* ou enviar um *Pull Request*.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](https://www.google.com/search?q=LICENSE) para mais detalhes.
