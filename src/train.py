"""Train a generative QA model using the dataset created with generate_qa_dataset.py"""

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


def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "question" in item and "answer" in item:
                data.append(item)
    return Dataset.from_list(data)


def preprocess_examples(examples, tokenizer, prefix="question: "):
    inputs = [prefix + q for q in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)
    labels = tokenizer(text_target=examples["answer"], max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    print(f"Carregando dataset de '{INPUT_DATASET_FILE}'...")
    dataset = load_dataset(INPUT_DATASET_FILE)
    print(f"{len(dataset)} pares de P&R carregados.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

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
    print("Métricas de avaliação:", metrics)


if __name__ == "__main__":
    main()