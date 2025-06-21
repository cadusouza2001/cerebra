"""Simple evaluation script for the trained model."""

# Avalia rapidamente o modelo de QA já treinado. Selecionamos algumas
# perguntas do dataset de validação e verificamos qual resposta o modelo
# gera, comparando com a resposta esperada.

import os
import json
import random
from transformers import pipeline

DATASET_FILE = os.getenv("DATASET_FILE", "qa_dataset/spark_qa_generative_dataset.jsonl")
MODEL_DIR = os.getenv("MODEL_DIR", "spark_expert_model")
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 5))

# NUM_SAMPLES controla quantos exemplos aleatórios iremos inspecionar.


def load_samples(path, num_samples):
    data = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    # Selecionamos aleatoriamente 'num_samples' pares de P&R para a avaliação
    return random.sample(data, min(num_samples, len(data)))


def main():
    samples = load_samples(DATASET_FILE, NUM_SAMPLES)
    # Usamos a tarefa de text2text-generation pois o modelo é do tipo
    # seq2seq (encoder-decoder), treinado para mapear perguntas em respostas.
    generator = pipeline("text2text-generation", model=MODEL_DIR, tokenizer=MODEL_DIR)
    # A pipeline encapsula o modelo seq2seq e a tokenização. Aqui usamos
    # o mesmo checkpoint gerado em train.py.

    for sample in samples:
        question = sample["question"]
        expected = sample["answer"]
        prediction = generator(question, max_new_tokens=128)[0]["generated_text"]
        # Mostramos lado a lado pergunta, gabarito (ground truth) e
        # a resposta gerada para que possamos avaliar qualitativamente a
        # performance do modelo.
        print("Q:", question)
        print("GT:", expected)
        print("Pred:", prediction)
        print("-" * 40)


if __name__ == "__main__":
    main()
