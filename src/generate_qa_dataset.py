# generate_qa_dataset.py (Versão com a Correção Final das Chaves do JSON)

import os
import json
import time
import random
import google.generativeai as genai

# --- Configurações ---
INPUT_DATA_FILE = "spark_docs_scrape/spark_documentation_guides.jsonl" 
OUTPUT_FILE = "qa_dataset/spark_qa_dataset_final.jsonl"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Configuração da API Gemini ---
try:
    genai.configure(api_key="<key>")
except Exception as e:
    print("Chave de API não configurada.")
    exit()

def create_chunks(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide um texto longo em pedaços menores e sobrepostos."""
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def call_llm_for_qa(text_chunk):
    """Faz uma chamada de API para gerar um par de P&R."""
    # Usando o modelo Gemma que você confirmou que é válido
    model = genai.GenerativeModel("gemma-3-27b-it") 
    prompt = f"""
    Sua tarefa é analisar o texto técnico fornecido e gerar exatamente um par de pergunta e resposta de alta qualidade.
    RESTRIÇÕES CRÍTICAS:
    1. A 'answer_text' DEVE SER uma substring exata e contígua do texto original.
    2. Sua resposta deve ser APENAS um único objeto JSON válido com as chaves "question_text" e "answer_text".
    Texto: "{text_chunk}"
    Sua Resposta JSON:
    """
    try:
        request_options = {"timeout": 100}
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.3))
        
        if not response.parts:
            print(f"  └─ [AVISO] Resposta da API bloqueada ou vazia. Feedback: {response.prompt_feedback}")
            return None
            
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        qa_pair = json.loads(json_text)
        
        # --- CORREÇÃO 1: VERIFICAR A CHAVE CORRETA ---
        # Verificamos por "question_text" em vez de "question".
        if "question_text" in qa_pair and "answer_text" in qa_pair:
            return qa_pair
        else:
            print(f"  └─ [AVISO] JSON retornado não continha as chaves esperadas. Recebido: {qa_pair}")
            return None
            
    except Exception as e:
        print(f"  └─ [ERRO] Ocorreu um erro na chamada da API: {e}")
        return None

# --- Lógica principal do script ---

print(f"Carregando documentação do arquivo '{INPUT_DATA_FILE}'...")
scraped_pages = []
try:
    with open(INPUT_DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            scraped_pages.append(json.loads(line))
except FileNotFoundError:
    print(f"[ERRO] Arquivo de entrada '{INPUT_DATA_FILE}' não encontrado!")
    exit()

print("Dividindo o conteúdo das páginas em pedaços (chunks)...")
all_chunks = []
for page in scraped_pages:
    if page.get("content"):
        page_chunks = create_chunks(page["content"])
        all_chunks.extend(page_chunks)
print(f"Gerados {len(all_chunks)} pedaços de texto para processamento.")

# Para rodar em todos os dados, use a linha abaixo
chunks_to_process = all_chunks
# Para um teste rápido, descomente a linha abaixo e comente a de cima
# chunks_to_process = all_chunks[:50] 

with open(OUTPUT_FILE, 'w') as f: pass

SECONDS_PER_REQUEST = 4 
print(f"Processando {len(chunks_to_process)} pedaços com a API (esperando {SECONDS_PER_REQUEST}s entre chamadas)...")
success_counter = 0
for i, chunk in enumerate(chunks_to_process):
    print(f"\n--- Processando chunk {i + 1}/{len(chunks_to_process)} ---")
    time.sleep(SECONDS_PER_REQUEST)
    
    qa_pair = call_llm_for_qa(chunk)
    
    if qa_pair:
        answer_start = chunk.find(qa_pair["answer_text"])
        if answer_start != -1:
            # --- CORREÇÃO 2: USAR A CHAVE CORRETA AO SALVAR ---
            squad_item_to_save = {
                "context": chunk,
                "question": qa_pair["question_text"], # Usamos a chave correta aqui
                "answers": {"text": [qa_pair["answer_text"]], "answer_start": [answer_start]}
            }
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(squad_item_to_save, ensure_ascii=False) + '\n')
            success_counter += 1
            print(f"  └─ [SUCESSO] Par de P&R salvo. ({success_counter} no total)")
        else:
            print(f"  └─ [AVISO] Resposta da API ('{qa_pair['answer_text']}') não encontrada no contexto original.")

print(f"\nProcesso concluído. Gerados e salvos {success_counter} pares de P&R no arquivo '{OUTPUT_FILE}'.")