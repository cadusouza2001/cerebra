import json
from collections import Counter
from typing import List, Tuple

import torch
from torch import nn

# Este arquivo implementa os blocos fundamentais do nosso modelo
# de Perguntas e Respostas em PyTorch. Aqui aparecem conceitos
# essenciais da disciplina, como tokenização, vetorização e a
# arquitetura encoder-decoder baseada em LSTMs.


def simple_tokenize(text: str) -> List[str]:
    """Very basic tokenizer: lowercase and split on spaces."""
    # A tokenização transforma o texto em "pedacinhos" (tokens).
    # É o primeiro passo de qualquer pipeline de NLP para que a
    # rede consiga lidar com palavras em formato numérico.
    return text.lower().split()


class Vocab:
    """Maps tokens to integer ids and back."""

    # Essa classe implementa uma "vetorização" bem simples: cada
    # palavra única recebe um índice inteiro. É o mesmo princípio
    # dos embeddings que estudamos, mas aqui de forma manual.

    def __init__(self, tokens: List[str], min_freq: int = 1):
        counter = Counter(tokens)
        self.itos = ["<pad>", "<unk>", "<bos>", "<eos>"]
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self.itos:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, tokens: List[str]) -> List[int]:
        # Converte tokens em índices inteiros. Tokens desconhecidos
        # viram <unk>, algo importante para lidar com palavras raras.
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]

    def decode(self, indices: List[int]) -> str:
        # Faz o caminho inverso: de índices para palavras.
        words = []
        for idx in indices:
            if idx == self.stoi["<eos>"]:
                break
            words.append(self.itos[idx])
        return " ".join(words)

    @property
    def pad_index(self) -> int:
        # Usado para preencher (padding) sequências menores que o máximo.
        return self.stoi["<pad>"]

    @property
    def bos_index(self) -> int:
        # "Beginning of sequence" para o decoder iniciar a geração.
        return self.stoi["<bos>"]

    @property
    def eos_index(self) -> int:
        # Indica o fim da sequência gerada.
        return self.stoi["<eos>"]


class QADataset(torch.utils.data.Dataset):
    """Loads a QA jsonl dataset and converts to indices."""

    # Representa nosso dataset supervisionado. Cada item contém
    # a pergunta e a resposta já tokenizadas e convertidas em
    # índices numéricos, prontos para a rede.

    def __init__(self, path: str, vocab: Vocab = None):
        # Abrimos o arquivo .jsonl gerado anteriormente
        data = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.questions = [simple_tokenize(d["question"]) for d in data]
        self.answers = [simple_tokenize(d["answer"]) for d in data]

        if vocab is None:
            # Construímos o vocabulário a partir de todas as palavras
            # presentes no dataset.
            tokens = []
            for q, a in zip(self.questions, self.answers):
                tokens.extend(q)
                tokens.extend(a)
            self.vocab = Vocab(tokens)
        else:
            self.vocab = vocab

        # Perguntas e respostas são convertidas em sequências de índices.
        self.q_indices = [self.vocab.encode(q) for q in self.questions]
        self.a_indices = [self.vocab.encode(a) for a in self.answers]

    def __len__(self) -> int:
        # Quantidade de pares pergunta→resposta no dataset.
        return len(self.q_indices)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        # Retorna os índices já vetorizados para DataLoader.
        return self.q_indices[idx], self.a_indices[idx]


def collate_batch(batch, pad_index: int, bos_index: int, eos_index: int):
    """Pad sequences and add BOS/EOS tokens for decoder input."""
    # Função auxiliar usada pelo DataLoader para juntar exemplos de
    # tamanhos diferentes em um único tensor (padding). Também
    # adicionamos <bos> e <eos> para o decoder saber onde começar e
    # terminar a geração.
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

    # Esta rede ilustra a arquitetura de um modelo seq2seq
    # que vimos em aula. O encoder lê a pergunta e gera um
    # vetor de estado. O decoder usa esse estado para gerar a
    # resposta palavra por palavra.

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        # Camada de embeddings: transforma cada índice de palavra em
        # um vetor denso, aprendendo representações semânticas.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # LSTM é um tipo de RNN capaz de lidar com sequências longas.
        # Aqui usamos uma para codificar a pergunta e outra para decodificar
        # a resposta.
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # Camada linear final que "transforma" o hidden state em
        # probabilidades de cada palavra do vocabulário.
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass used for training."""
        # 1. Codificamos a pergunta (src) e capturamos o estado final
        enc_emb = self.embedding(src)
        _, (hidden, cell) = self.encoder(enc_emb)
        # 2. O decoder recebe a resposta de treinamento deslocada
        # (teacher forcing) para prever o próximo token a cada passo
        dec_emb = self.embedding(tgt[:, :-1])
        outputs, _ = self.decoder(dec_emb, (hidden, cell))
        # 3. Camada linear gera as distribuições de probabilidade
        # para cada posição da sequência
        logits = self.fc(outputs)
        return logits

    def generate(self, src: torch.Tensor, bos_index: int, eos_index: int, max_len: int = 50) -> List[int]:
        """Greedy decoding for inference."""
        # Modo de geração: usamos o próprio modelo para prever token
        # a token até encontrar <eos> ou atingir o limite de passos.
        self.eval()
        generated = []
        with torch.no_grad():
            enc_emb = self.embedding(src)
            _, (hidden, cell) = self.encoder(enc_emb)
            inp = torch.tensor([[bos_index]], device=src.device)
            for _ in range(max_len):
                dec_emb = self.embedding(inp)
                output, (hidden, cell) = self.decoder(dec_emb, (hidden, cell))
                logits = self.fc(output[:, -1])
                pred = logits.argmax(1)
                generated.append(pred.item())
                if pred.item() == eos_index:
                    break
                inp = pred.unsqueeze(1)
        return generated
