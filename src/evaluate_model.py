"""Simple evaluation script for the trained model."""

import os
import json
import random
from transformers import pipeline

DATASET_FILE = os.getenv("DATASET_FILE", "qa_dataset/spark_qa_generative_dataset.jsonl")
MODEL_DIR = os.getenv("MODEL_DIR", "spark_expert_model")
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 5))


def load_samples(path, num_samples):
    data = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    return random.sample(data, min(num_samples, len(data)))


def main():
    samples = load_samples(DATASET_FILE, NUM_SAMPLES)
    generator = pipeline("text2text-generation", model=MODEL_DIR, tokenizer=MODEL_DIR)

    for sample in samples:
        question = sample["question"]
        expected = sample["answer"]
        prediction = generator(question, max_new_tokens=128)[0]["generated_text"]
        print("Q:", question)
        print("GT:", expected)
        print("Pred:", prediction)
        print("-" * 40)


if __name__ == "__main__":
    main()
