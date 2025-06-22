import os
import random
import torch

# Avalia o desempenho do modelo treinado em algumas amostras do dataset.
# A ideia é verificar qualitativamente se ele aprendeu a gerar respostas
# coerentes, usando os conceitos de métricas e validação.

from qa_model import QADataset, Seq2SeqModel, Vocab

DATASET_FILE = os.getenv("DATASET_FILE", "qa_dataset/spark_qa_generative_dataset.jsonl")
MODEL_DIR = os.getenv("MODEL_DIR", "spark_expert_model")
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 5))


def load_checkpoint(model_dir):
    # Carregamos o modelo treinado e seu vocabulário. É o mesmo
    # checkpoint salvo durante a fase de treinamento.
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    vocab = Vocab([])
    vocab.itos = ckpt["vocab"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    model = Seq2SeqModel(len(vocab.itos))
    model.load_state_dict(ckpt["model_state"])
    return model, vocab


def main():
    """Avalia o modelo simples de QA treinado com PyTorch."""
    model, vocab = load_checkpoint(MODEL_DIR)
    model.eval()
    dataset = QADataset(DATASET_FILE, vocab)

    # Selecionamos aleatoriamente algumas amostras para inspecionar
    # as respostas geradas. É uma forma simples de avaliação
    # qualitativa, além das métricas automáticas vistas em aula.

    samples = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for idx in samples:
        q_idxs = dataset[idx]
        question = " ".join(dataset.questions[idx])
        expected = " ".join(dataset.answers[idx])

        q_tensor = torch.tensor([q_idxs], dtype=torch.long, device=device)
        # Geração da resposta pelo modelo, assim como fazemos na fase
        # de inferência do sistema.
        pred_idxs = model.generate(q_tensor, vocab.bos_index, vocab.eos_index)
        prediction = vocab.decode(pred_idxs)

        print("Q:", question)
        print("GT:", expected)
        print("Pred:", prediction)
        print("-" * 40)


if __name__ == "__main__":
    main()
