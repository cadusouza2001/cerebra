"""Pequeno sistema interativo para fazer perguntas ao modelo treinado."""

import os
import torch

from qa_model import Seq2SeqModel, Vocab, simple_tokenize


class SparkQASystem:
    def __init__(self, model_dir: str):
        ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        self.vocab = Vocab([])
        self.vocab.itos = ckpt["vocab"]
        self.vocab.stoi = {t: i for i, t in enumerate(self.vocab.itos)}
        self.model = Seq2SeqModel(len(self.vocab.itos))
        self.model.load_state_dict(ckpt["model_state"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def ask(self, question: str) -> str:
        tokens = simple_tokenize(question)
        idxs = [self.vocab.stoi.get(t, self.vocab.stoi["<unk>"]) for t in tokens]
        src = torch.tensor([idxs], dtype=torch.long, device=self.device)
        pred = self.model.generate(src, self.vocab.bos_index, self.vocab.eos_index)
        return self.vocab.decode(pred)


if __name__ == "__main__":
    system = SparkQASystem(model_dir="spark_expert_model")
    questions = [
        "What is Apache Spark?",
        "How to cache a Dataset?",
    ]
    for q in questions:
        ans = system.ask(q)
        print(f"\n{q}\n-> {ans}")
