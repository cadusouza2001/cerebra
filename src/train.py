"""Treina um modelo simples de Pergunta e Resposta utilizando PyTorch puro."""

# Esta versão não usa Transformers pré-treinados. Construímos do zero uma
# rede encoder-decoder pequena, baseada em embeddings e LSTMs. O objetivo é
# demonstrar os conceitos de redes neurais aplicados a NLP e treinamento
# supervisionado.

import os
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from qa_model import QADataset, collate_batch, Seq2SeqModel

# Caminhos configuráveis por variáveis de ambiente
INPUT_DATASET_FILE = os.getenv("DATASET_FILE", "qa_dataset/spark_qa_generative_dataset.jsonl")
OUTPUT_MODEL_DIR = os.getenv("OUTPUT_MODEL_DIR", "spark_expert_model")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))


def main():
    print(f"Carregando dataset de '{INPUT_DATASET_FILE}'...")
    dataset = QADataset(INPUT_DATASET_FILE)
    vocab = dataset.vocab
    print(f"{len(dataset)} pares de perguntas e respostas carregados.")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, vocab.pad_index, vocab.bos_index, vocab.eos_index),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Seq2SeqModel(len(vocab.itos)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for questions, answers in dataloader:
            questions = questions.to(device)
            answers = answers.to(device)

            optimizer.zero_grad()
            logits = model(questions, answers)
            loss = criterion(logits.reshape(-1, logits.size(-1)), answers[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - perda média: {avg_loss:.4f}")

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    save_path = Path(OUTPUT_MODEL_DIR) / "model.pt"
    torch.save({"model_state": model.state_dict(), "vocab": vocab.itos}, save_path)
    print(f"Modelo salvo em {save_path}")


if __name__ == "__main__":
    main()
