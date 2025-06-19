# generate_qa_dataset.py

import os
import json
import time
import random
import google.generativeai as genai
from txtai.embeddings import Embeddings

# --- Implementação da API Gemini Pro ---
# IMPORTANTE: Substitua "YOUR_API_KEY_HERE" pela sua chave de API Gemini real
try:
    genai.configure(api_key="<api-key>")
except Exception as e:
    print(
        "Chave de API não configurada. Por favor, substitua 'YOUR_API_KEY_HERE' pela sua chave real."
    )
    exit()


def call_gemini_for_qa(text_chunk):
    """
    Faz uma chamada de API real para o Google Gemini Pro para gerar um par de P&R.
    """
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"""
    Você é um assistente especialista projetado para criar pares de perguntas e respostas de alta qualidade para um conjunto de dados de treinamento de aprendizado de máquina.
    Sua tarefa é analisar o texto fornecido da documentação do Apache Spark e gerar exatamente uma pergunta de alta qualidade que possa ser respondida a partir do texto.

    RESTRIÇÕES CRÍTICAS:
    1. A 'answer_text' que você fornecer DEVE SER uma substring exata e contígua do texto original. Não reformule ou invente uma resposta.
    2. Sua resposta deve ser APENAS um único objeto JSON válido. Não inclua nenhum outro texto, explicações ou formatação markdown como ```json.

    EXEMPLO:
    Texto: "A propriedade spark.driver.memory controla a quantidade de memória a ser usada para o processo do driver. O padrão é 1g."
    Sua Resposta JSON:
    {{
      "question": "Qual é o valor padrão para a propriedade spark.driver.memory?",
      "answer_text": "1g"
    }}

    Agora, por favor, processe o seguinte texto:
    ---
    Texto: "{text_chunk}"
    ---
    Sua Resposta JSON:
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.4),
            safety_settings={
                "HATE": "BLOCK_NONE",
                "HARASSMENT": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE",
            },
        )
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        qa_pair = json.loads(json_text)
        if "question" in qa_pair and "answer_text" in qa_pair:
            return qa_pair
        return None
    except Exception as e:
        print(f"Ocorreu um erro durante a chamada da API: {e}")
        return None


# --- Lógica principal do script ---

print("Carregando pedaços indexados para gerar perguntas...")
embeddings = Embeddings()
embeddings.load("spark_docs.index")
all_chunks = [item["text"] for item in embeddings.search(" ", limit=len(embeddings))]
print(f"Carregados {len(all_chunks)} pedaços totais.")

if len(all_chunks) > 200:
    chunks_to_process = random.sample(all_chunks, 200)
else:
    chunks_to_process = all_chunks

print(f"Processando {len(chunks_to_process)} pedaços com a API Gemini...")
squad_formatted_data = []
for i, chunk in enumerate(chunks_to_process):
    time.sleep(1)  # Dorme por 1 segundo entre cada chamada de API
    qa_pair = call_gemini_for_qa(chunk)
    if qa_pair:
        answer_start = chunk.find(qa_pair["answer_text"])
        if answer_start != -1:
            squad_formatted_data.append(
                {
                    "context": chunk,
                    "question": qa_pair["question"],
                    "answers": {
                        "text": [qa_pair["answer_text"]],
                        "answer_start": [answer_start],
                    },
                }
            )
            print(f"Processado com sucesso {i + 1}/{len(chunks_to_process)}")
        else:
            print(
                f"Aviso: Não foi possível encontrar a resposta '{qa_pair['answer_text']}' no contexto. Pulando."
            )

with open("spark_qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(squad_formatted_data, f, indent=4, ensure_ascii=False)

print(f"\nGerados {len(squad_formatted_data)} pares de P&R.")
print("Conjunto de dados salvo em spark_qa_dataset.json")

