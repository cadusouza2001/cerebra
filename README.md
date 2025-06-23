# 1. Cerebra 🧠

## 1.1. Visão Geral do Projeto

O **Cerebra** é um sistema simples de Perguntas e Respostas (Q&A) construído em Python. Ele transforma uma coleção de páginas da web em um especialista capaz de responder a questões sobre aquele conteúdo. Todo o modelo é treinado do zero utilizando PyTorch e dados gerados automaticamente com a API Gemini.

## 1.2. Descrição do Problema e Justificativa

Sistemas de Perguntas e Respostas (Q&A) são cada vez mais utilizados em aplicações que exigem consulta a documentações técnicas, como bases de conhecimento, manuais ou APIs. No entanto, esses sistemas frequentemente exigem treinamento caro ou não são adaptáveis a domínios específicos.

Este projeto propõe uma solução que transforma automaticamente uma documentação técnica (neste caso, a do Apache Spark) em um modelo especialista capaz de responder perguntas com base no conteúdo. O sistema utiliza redes neurais para entender linguagem natural, sendo útil em contextos educacionais, técnicos e corporativos.

**Justificativa do uso de redes neurais:**  
Redes neurais oferecem flexibilidade para capturar padrões linguísticos e gerar respostas coesas. O modelo foi treinado do zero, reforçando os conceitos da disciplina e permitindo controle total da arquitetura.

## 1.3. Sobre o Dataset

O conjunto de dados foi gerado automaticamente a partir da documentação do Apache Spark usando a API Gemini. Cada entrada é composta por uma **pergunta gerada automaticamente** e uma **resposta contextualizada**, com base em trechos extraídos da documentação oficial.

- **Formato:** JSONL com pares `{"question": ..., "answer": ...}`
- **Tamanho aproximado:** 12.000 pares
- **Pré-processamento:** chunking com sobreposição de texto e verificação de coerência da resposta com o trecho original
- **Local:** `qa_dataset/spark_qa_generative_dataset.jsonl`

> O processo de geração e validação do dataset está descrito no script `generate_qa_dataset.py`.

## 1.4. Arquitetura do Modelo

A rede neural utilizada para geração de respostas é um modelo **seq2seq** (codificador-decodificador) treinado com PyTorch. O modelo foi criado do zero no script `qa_model.py` e treinado com o script `train.py`.
Como o volume de dados é limitado, também disponibilizamos um exemplo de **fine-tuning** de um modelo pré-treinado (`t5-small`) no script `fine_tune_t5.py`.
Essa abordagem reutiliza o conhecimento de um modelo grande e reduz o tempo de treinamento.

- **Número de camadas:** 2 camadas LSTM (encoder e decoder)
- **Função de ativação:** tanh (implícita nas células LSTM)
- **Embedding:** matriz de embedding treinável (não pré-treinada)
- **Perda utilizada:** CrossEntropyLoss
- **Otimização:** Adam

O foco está no **componente Reader**, responsável por formular a resposta com base na pergunta, enquanto a busca de documentos relevantes é feita pelo `txtai` (Retriever).

### 1.4.1. Tecnologias e bibliotecas

- Python 3.9+
- `aiohttp` e `BeautifulSoup4` para raspagem
- `txtai` para busca por similaridade (Retriever)
- `torch` para treinar a rede neural
- `google-generativeai` para criar pares de pergunta e resposta

## 1.5. Indexação e Recuperação

O script `index_spark_docs.py` converte cada página da documentação em um vetor
semântico usando `txtai`. O índice resultante é salvo em `spark_docs.index` e,
com a opção `content=True`, o texto original fica acessível para buscas.
Durante a inferência, `run_qa_system.py` carrega esse índice e utiliza os
trechos recuperados como contexto para o modelo.

### 1.6. Fazendo Perguntas

Com o índice criado e o modelo treinado, é possível fazer perguntas via linha de
comando passando a questão diretamente para `run_qa_system.py`:

```bash
python src/run_qa_system.py "What is Apache Spark?"
```

# 2. Execução no Google Colab

Abra o notebook [cerebra_pipeline.ipynb](./cerebra_pipeline.ipynb) no Colab e execute as células em ordem. Ele contém todos os comandos necessários para instalar dependências, configurar as variáveis de ambiente, raspar a documentação, treinar o modelo e realizar as avaliações.
