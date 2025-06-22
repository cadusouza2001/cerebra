import json
from collections import Counter
from typing import List, Tuple

import torch
from torch import nn


def simple_tokenize(text: str) -> List[str]:
    """Very basic tokenizer: lowercase and split on spaces."""
    return text.lower().split()


class Vocab:
    """Maps tokens to integer ids and back."""

    def __init__(self, tokens: List[str], min_freq: int = 1):
        counter = Counter(tokens)
        self.itos = ["<pad>", "<unk>", "<bos>", "<eos>"]
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self.itos:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]

    def decode(self, indices: List[int]) -> str:
        words = []
        for idx in indices:
            if idx == self.stoi["<eos>"]:
                break
            words.append(self.itos[idx])
        return " ".join(words)

    @property
    def pad_index(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_index(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_index(self) -> int:
        return self.stoi["<eos>"]


class QADataset(torch.utils.data.Dataset):
    """Loads a QA jsonl dataset and converts to indices."""

    def __init__(self, path: str, vocab: Vocab = None):
        data = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.questions = [simple_tokenize(d["question"]) for d in data]
        self.answers = [simple_tokenize(d["answer"]) for d in data]

        if vocab is None:
            tokens = []
            for q, a in zip(self.questions, self.answers):
                tokens.extend(q)
                tokens.extend(a)
            self.vocab = Vocab(tokens)
        else:
            self.vocab = vocab

        self.q_indices = [self.vocab.encode(q) for q in self.questions]
        self.a_indices = [self.vocab.encode(a) for a in self.answers]

    def __len__(self) -> int:
        return len(self.q_indices)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.q_indices[idx], self.a_indices[idx]


def collate_batch(batch, pad_index: int, bos_index: int, eos_index: int):
    """Pad sequences and add BOS/EOS tokens for decoder input."""
    qs, ans = zip(*batch)
    q_max = max(len(q) for q in qs)
    a_max = max(len(a) for a in ans)

    q_tensor = torch.full((len(batch), q_max), pad_index, dtype=torch.long)
    a_tensor = torch.full((len(batch), a_max + 2), pad_index, dtype=torch.long)
    for i, (q, a) in enumerate(zip(qs, ans)):
        q_tensor[i, : len(q)] = torch.tensor(q, dtype=torch.long)
        a_tensor[i, 0] = bos_index
        a_tensor[i, 1 : len(a) + 1] = torch.tensor(a, dtype=torch.long)
        a_tensor[i, len(a) + 1] = eos_index
    return q_tensor, a_tensor


class Seq2SeqModel(nn.Module):
    """Minimal encoder-decoder network with embeddings and LSTMs."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Embedding layer shared by encoder and decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Dropout ajuda a regularizar o modelo evitando overfitting
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass used for training."""
        # Aplicamos dropout logo após os embeddings para regularização
        enc_emb = self.dropout(self.embedding(src))
        _, (hidden, cell) = self.encoder(enc_emb)
        dec_emb = self.dropout(self.embedding(tgt[:, :-1]))
        outputs, _ = self.decoder(dec_emb, (hidden, cell))
        logits = self.fc(outputs)
        return logits

    def generate(self, src: torch.Tensor, bos_index: int, eos_index: int, max_len: int = 50) -> List[int]:
        """Greedy decoding for inference."""
        self.eval()
        generated = []
        with torch.no_grad():
            enc_emb = self.dropout(self.embedding(src))
            _, (hidden, cell) = self.encoder(enc_emb)
            inp = torch.tensor([[bos_index]], device=src.device)
            for _ in range(max_len):
                dec_emb = self.dropout(self.embedding(inp))
                output, (hidden, cell) = self.decoder(dec_emb, (hidden, cell))
                logits = self.fc(output[:, -1])
                pred = logits.argmax(1)
                generated.append(pred.item())
                if pred.item() == eos_index:
                    break
                inp = pred.unsqueeze(1)
        return generated
