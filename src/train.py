# Célula de Treinamento - Cole e execute no Google Colab com ambiente de GPU

# 1. Instala as dependências necessárias
print("Instalando dependências...")
!pip install transformers datasets torch accelerate sentencepiece -q
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
print("Dependências prontas.")

# 2. Importa as bibliotecas
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# --- Nomes de Arquivos e Diretórios (no seu Google Drive) ---
# O novo dataset generativo que você acabou de criar
INPUT_DATASET_FILE = "/content/drive/MyDrive/Unisinos/Cerebra/spark_qa_generative_dataset.jsonl" 
# Onde o novo modelo generativo será salvo
OUTPUT_MODEL_DIR = "/content/drive/MyDrive/Unisinos/Cerebra/spark_generative_expert_model" 

# --- 3. Carrega o seu novo dataset generativo ---
print(f"Carregando o dataset generativo de '{INPUT_DATASET_FILE}'...")
raw_data = []
try:
    with open(INPUT_DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            # O formato agora é mais simples: 'question' e 'answer'
            data_point = json.loads(line)
            if 'question' in data_point and 'answer' in data_point:
                 raw_data.append(data_point)
except FileNotFoundError:
    print(f"[ERRO] Arquivo de entrada '{INPUT_DATASET_FILE}' não encontrado!")
    print("Execute o script 'generate_qa_dataset...' primeiro.")
    raise

print(f"Carregados {len(raw_data)} pares de P&R para o treinamento generativo.")
dataset = Dataset.from_list(raw_data)

# --- 4. Carrega o Modelo e Tokenizador Texto-para-Texto ---
print("Carregando o modelo base (google/flan-t5-base)...")
# FLAN-T5 é um excelente modelo para tarefas generativas e de instrução.
model_checkpoint = "google/flan-t5-base" 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# --- 5. Pré-processa os Dados para o formato Texto-para-Texto ---
print("Pré-processando os dados para o formato texto-para-texto...")
# Definimos um prefixo para a tarefa, que ajuda o modelo a entender o que fazer.
prefix = "answer the question based on the context: "
max_input_length = 1024
max_target_length = 256

def preprocess_function(examples):
    # Formata a entrada: "prefixo + pergunta: [PERGUNTA] contexto: [CONTEXTO]"
    inputs = [prefix + "question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]
    
    # O alvo (label) é simplesmente a resposta gerada.
    targets = [ans for ans in examples["answer"]]
    
    # Tokeniza as entradas e os alvos
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
print("Pré-processamento concluído.")

# --- 6. Ajuste Fino do Modelo Generativo ---
print("Configurando o treinamento Seq2Seq...")

# Data Collator para agrupar e preencher os dados em lotes
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Argumentos de treinamento específicos para modelos Seq2Seq
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4, # Batch size menor é comum para modelos maiores
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True, # Acelera o treino em GPUs
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset, # Avaliamos no mesmo dataset por simplicidade
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("\n" + "="*50)
print("INICIANDO O TREINAMENTO DO MODELO GENERATIVO")
print