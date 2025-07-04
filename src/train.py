
# ------------------------------------------------------
# Este script mostra na prática vários conceitos vistos na
# disciplina de Inteligência Artificial e Redes Neurais.
# Ele monta uma rede neural do tipo *encoder-decoder* com
# embeddings e LSTMs e a treina de forma supervisionada.
# Dessa forma, conseguimos ilustrar passo a passo o que
# acontece em um treinamento de rede para NLP.
# ------------------------------------------------------

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from qa_model import QADataset, collate_batch, Seq2SeqModel

# ------------------------------------------------------------------
# QADataset carrega o conjunto de exemplos Pergunta→Resposta.  Essa
# é nossa base rotulada para "treinamento supervisionado".
# collate_batch cuida do empacotamento em lotes (batches), útil
# quando falamos de mini-batch gradient
# descent.
# Seq2SeqModel é a rede neural propriamente dita (um encoder-decoder
# com LSTMs).  Cada componente será criado logo abaixo.
# ------------------------------------------------------------------

# ------------------------------------------------------
# Caminhos configuráveis por variáveis de ambiente.
# Facilita alterar o dataset ou o local de saída sem
# modificar o código.  Algo útil quando fazemos vários
# experimentos durante o treinamento.
# ------------------------------------------------------
INPUT_DATASET_FILE = os.getenv("DATASET_FILE", "qa_dataset/spark_qa_generative_dataset.jsonl")
OUTPUT_MODEL_DIR = os.getenv("OUTPUT_MODEL_DIR", "spark_expert_model")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 1000))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LOG_FILE = os.getenv("TRAINING_LOG", "training_log.csv")


def main():
    """Treina um modelo simples de Pergunta e Resposta utilizando PyTorch puro."""
    print(f"Carregando dataset de '{INPUT_DATASET_FILE}'...")
    # Aqui entra o conceito de **dataset rotulado**. Cada linha do
    # arquivo contém uma pergunta e a resposta correta que o modelo
    # deve aprender a reproduzir.
    dataset = QADataset(INPUT_DATASET_FILE)
    vocab = dataset.vocab
    print(f"{len(dataset)} pares de perguntas e respostas carregados.")

    # Dividimos em treino e validação (80/20) para monitorar overfitting.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, vocab.pad_index, vocab.bos_index, vocab.eos_index),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, vocab.pad_index, vocab.bos_index, vocab.eos_index),
    )

    # DataLoader é responsável por dividir o dataset em *batches*.
    # Cada batch passa pela rede em uma etapa do gradiente, se baseando
    # em otimização com mini-batch.

    # Se houver GPU disponível, usamos CUDA. É a parte prática de
    # acelerar o treinamento das redes vistas em aula.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instanciamos nossa rede neural. Ela possui embeddings, um LSTM
    # encoder e um LSTM decoder, formando um modelo seq2seq.
    model = Seq2SeqModel(len(vocab.itos)).to(device)

    # A função de perda CrossEntropy combina o softmax com a entropia
    # cruzada, muito utilizada quando queremos treinar um classificador
    # ou, como aqui, prever o próximo token correto.
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_index)

    # Adam realiza o ajuste dos pesos com base no gradiente calculado
    # (o famoso backpropagation). A taxa de aprendizado padrão é 1e-3.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    best_val_loss = float("inf")
    # Se a perda de validação não melhorar por "patience" épocas,
    # encerramos o treino (early stopping)
    patience = 50
    epochs_no_improve = 0
    for epoch in range(NUM_EPOCHS):
        # Cada passada completa no dataset é uma *época* (epoch).
        model.train()
        total_loss = 0.0
        for questions, answers in train_loader:
            # Enviamos o lote para a CPU ou GPU
            questions = questions.to(device)
            answers = answers.to(device)

            optimizer.zero_grad()
            # Feedforward: passamos perguntas e respostas pela rede
            logits = model(questions, answers)
            # Calculamos a perda comparando a saída do modelo com a
            # resposta real. Note o uso do "ignore_index" para não
            # considerar o padding no cálculo.
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                answers[:, 1:].reshape(-1),
            )
            # Retropropagação: gradientes fluem para ajustar os pesos
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Avaliamos no conjunto de validação sem atualizar os pesos
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for q_val, a_val in val_loader:
                q_val = q_val.to(device)
                a_val = a_val.to(device)
                logits = model(q_val, a_val)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    a_val[:, 1:].reshape(-1),
                )
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        # Registramos as perdas para análise posterior de convergência
        history.append((epoch + 1, avg_loss, avg_val_loss))
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - perda treino: {avg_loss:.4f} - perda val: {avg_val_loss:.4f}"
        )
        # Salvamos o modelo apenas quando ocorre melhoria na perda de validação
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
            save_path = Path(OUTPUT_MODEL_DIR) / "model.pt"
            torch.save({"model_state": model.state_dict(), "vocab": vocab.itos}, save_path)
            print(f"Epoch {epoch + 1}: nova melhor val_loss {avg_val_loss:.4f}, modelo salvo em {save_path}")
        else:
            epochs_no_improve += 1

        # Checamos se a validação não melhora por várias épocas (early stopping)
        if epochs_no_improve >= patience:
            print(f"Early stopping acionado após {patience} épocas sem melhoria.")
            break

    # Exporta histórico de perdas para análise de gráficos.
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")
        for ep, tr, vl in history:
            f.write(f"{ep},{tr:.6f},{vl:.6f}\n")

    # Gera e salva um gráfico com as curvas de perda de treino e validação
    epochs, train_losses, val_losses = zip(*history)
    plt.figure()
    plt.plot(epochs, train_losses, label="treino")
    plt.plot(epochs, val_losses, label="validação")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.title("Curva de Perda")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_plot.png")


if __name__ == "__main__":
    # O script é executado diretamente. Em notebooks chamaríamos main()
    # para iniciar o processo de treinamento e acompanhar as épocas.
    main()
