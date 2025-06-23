import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments


class QADataset(Dataset):
    """Dataset simples para pares de pergunta e resposta."""

    def __init__(self, path: str, tokenizer):
        self.data = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Tokenizamos pergunta e resposta separadamente
        model_inputs = self.tokenizer(
            item["question"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        labels = self.tokenizer(
            item["answer"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs


def main():
    """Exemplo de fine-tuning usando um modelo T5 pré-treinado."""

    dataset_file = os.getenv("DATASET_FILE", "qa_dataset/spark_qa_generative_dataset.jsonl")
    model_name = os.getenv("MODEL_NAME", "t5-small")
    output_dir = os.getenv("OUTPUT_MODEL_DIR", "spark_t5_model")
    num_epochs = int(os.getenv("NUM_EPOCHS", 3))
    batch_size = int(os.getenv("BATCH_SIZE", 8))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = QADataset(dataset_file, tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
