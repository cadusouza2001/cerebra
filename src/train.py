
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
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 100))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))


def main():
    """Treina um modelo simples de Pergunta e Resposta utilizando PyTorch puro."""
    print(f"Carregando dataset de '{INPUT_DATASET_FILE}'...")
    # Aqui entra o conceito de **dataset rotulado**. Cada linha do
    # arquivo contém uma pergunta e a resposta correta que o modelo
    # deve aprender a reproduzir.
    dataset = QADataset(INPUT_DATASET_FILE)
    vocab = dataset.vocab
    print(f"{len(dataset)} pares de perguntas e respostas carregados.")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    for epoch in range(NUM_EPOCHS):
        # Cada passada completa no dataset é uma *época* (epoch).
        model.train()
        total_loss = 0.0
        for questions, answers in dataloader:
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

        # Acompanhamos a perda média para verificar se a rede está
        # convergindo ou se poderia sofrer de overfitting/underfitting.
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - perda média: {avg_loss:.4f}")

    # Por fim salvamos os pesos e o vocabulário para usar na inferência.
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    save_path = Path(OUTPUT_MODEL_DIR) / "model.pt"
    torch.save({"model_state": model.state_dict(), "vocab": vocab.itos}, save_path)
    print(f"Modelo salvo em {save_path}")


if __name__ == "__main__":
    # O script é executado diretamente. Em notebooks chamaríamos main()
    # para iniciar o processo de treinamento e acompanhar as épocas.
    main()
