"""Train a generative QA model using the dataset created with generate_qa_dataset.py"""

# Script responsável por ajustar um modelo seq2seq (baseado em Transformer)
# no conjunto de perguntas e respostas gerado previamente. Essa etapa de
# fine-tuning segue o paradigma de aprendizado supervisionado: o modelo vê
# a pergunta (entrada) e a resposta correta (rótulo) e tenta minimizá-la
# através da função de perda cross-entropy.

import os
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Paths can be configured via environment variables
INPUT_DATASET_FILE = os.getenv(
    "DATASET_FILE", "qa_dataset/spark_qa_generative_dataset.jsonl"
)
OUTPUT_MODEL_DIR = os.getenv("OUTPUT_MODEL_DIR", "spark_expert_model")
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "google/flan-t5-base")

# MODEL_CHECKPOINT define qual modelo base será fine-tuned. O Flan-T5
# já possui arquitetura encoder-decoder, ideal para tarefas de geração
# condicionada como QA.


def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "question" in item and "answer" in item:
                data.append(item)
    # Convertemos a lista em um objeto Dataset do Hugging Face para
    # facilitar o manuseio durante o treinamento.
    return Dataset.from_list(data)


def preprocess_examples(examples, tokenizer, prefix="question: "):
    # Prefixamos cada pergunta para ajudar o modelo a entender a tarefa,
    # prática comum em fine-tuning de modelos como o T5.
    inputs = [prefix + q for q in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)
    labels = tokenizer(text_target=examples["answer"], max_length=256, truncation=True)
    # O Trainer do Hugging Face espera que as labels estejam no campo "labels".
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    # Função principal do treinamento
    print(f"Carregando dataset de '{INPUT_DATASET_FILE}'...")
    dataset = load_dataset(INPUT_DATASET_FILE)
    print(f"{len(dataset)} pares de P&R carregados.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # Aplicamos o pré-processamento em todo o dataset (tokenização das
    # perguntas e das respostas). "batched=True" usa vetorização para
    # acelerar.

    tokenized = dataset.map(
        lambda ex: preprocess_examples(ex, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
    )

    # Seq2SeqTrainer cuida do loop de treinamento, aplicando otimização
    # (AdamW por padrão) e cálculo da perda de cross-entropy.

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n==============================")
    print("Iniciando treinamento...")
    trainer.train()

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    print("Salvando modelo em", OUTPUT_MODEL_DIR)
    trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

    metrics = trainer.evaluate()
    # Ao final do treinamento avaliamos o desempenho no conjunto de
    # validação, obtendo métricas como a perda. Valores menores indicam
    # que o modelo está reproduzindo bem as respostas do dataset.
    print("Métricas de avaliação:", metrics)


if __name__ == "__main__":
    main()