# generate_qa_dataset_generative.py (Nova Versão com Prompt Aprimorado)

import os
import json
import time
import random
import google.generativeai as genai

# --- Configurações ---
INPUT_DATA_FILE = "spark_docs_scrape/spark_guides_dataset_clean.jsonl" 
OUTPUT_FILE = "spark_qa_generative_dataset.jsonl"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

# --- Configuração da API Gemini ---
# A chave da API agora é lida da variável de ambiente GEMINI_API_KEY
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY não definida. Defina a variável de ambiente antes de executar o script."
    )

genai.configure(api_key=api_key)

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
    """Faz uma chamada de API com um prompt aprimorado para respostas generativas."""
    # Usando um modelo poderoso para a geração
    model = genai.GenerativeModel("gemma-3-27b-it")
    
    # --- PROMPT APRIMORADO E EM INGLÊS ---
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
        request_options = {"timeout": 100}
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.4))
        
        if not response.parts:
            print(f"  └─ [AVISO] Resposta da API bloqueada. Feedback: {response.prompt_feedback}")
            return None
            
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        qa_pair = json.loads(json_text)
        
        # A nova verificação simples: apenas se as chaves existem.
        if "question" in qa_pair and "answer" in qa_pair:
            return qa_pair
        else:
            print(f"  └─ [AVISO] JSON retornado não continha as chaves 'question' ou 'answer'. Recebido: {qa_pair}")
            return None
            
    except Exception as e:
        print(f"  └─ [ERRO] Ocorreu um erro na chamada da API: {e}")
        return None

# --- Lógica principal do script ---
print(f"Carregando documentação do arquivo '{INPUT_DATA_FILE}'...")
scraped_pages = []
with open(INPUT_DATA_FILE, 'r', encoding='utf-8') as f:
    for line in f: scraped_pages.append(json.loads(line))

print("Dividindo o conteúdo das páginas em pedaços (chunks)...")
all_chunks = []
for page in scraped_pages:
    if page.get("content"):
        page_chunks = create_chunks(page["content"])
        all_chunks.extend(page_chunks)
print(f"Gerados {len(all_chunks)} pedaços de texto para processamento.")

chunks_to_process = all_chunks
with open(OUTPUT_FILE, 'w') as f: pass

# Valor seguro para respeitar os limites da API
SECONDS_PER_REQUEST = 4 

print(f"Processando {len(chunks_to_process)} pedaços com a API...")
success_counter = 0
for i, chunk in enumerate(chunks_to_process):
    print(f"\n--- Processando chunk {i + 1}/{len(chunks_to_process)} ---")
    time.sleep(SECONDS_PER_REQUEST)
    
    qa_pair = call_llm_for_qa(chunk)
    
    if qa_pair:
        # A lógica de verificação 'find' foi removida, pois não é mais necessária
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
        success_counter += 1
        print(f"  └─ [SUCESSO] Par de P&R generativo salvo. ({success_counter} no total)")

print(f"\nProcesso concluído. Gerados e salvos {success_counter} pares de P&R no arquivo '{OUTPUT_FILE}'.")