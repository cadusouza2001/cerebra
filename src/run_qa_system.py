import os
import torch
from txtai.embeddings import Embeddings

from qa_model import Seq2SeqModel, Vocab, simple_tokenize

# Aqui usamos o modelo seq2seq já treinado para responder novas
# perguntas. É a fase de inferência de um modelo de NLP.
class SparkQASystem:
    def __init__(self, model_dir: str, index_path: str = "spark_docs.index", top_k: int = 3):
        # Carregamos o checkpoint salvo após o treinamento.
        ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        self.vocab = Vocab([])
        self.vocab.itos = ckpt["vocab"]
        self.vocab.stoi = {t: i for i, t in enumerate(self.vocab.itos)}
        # A arquitetura precisa ser a mesma usada no treino.
        self.model = Seq2SeqModel(len(self.vocab.itos))
        self.model.load_state_dict(ckpt["model_state"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        # Carrega o índice vetorial previamente criado pelo txtai
        self.retriever = Embeddings()
        if os.path.exists(index_path):
            self.retriever.load(index_path)
        else:
            raise FileNotFoundError(f"Index path '{index_path}' not found")

        self.top_k = top_k

    def ask(self, question: str) -> str:
        # Primeiro, recuperamos os trechos de documentação mais relevantes.
        # Dependendo de como o índice foi criado, os resultados podem ser
        # dicionários com o texto ou apenas pares (id, score). Tratamos ambas
        # as situações para evitar erros de índice.
        results = self.retriever.search(question, self.top_k)
        context_parts = []
        if results and isinstance(results[0], dict) and "text" in results[0]:
            context_parts = [r["text"] for r in results]
        # Quando o índice não contém o texto original, seguimos sem contexto
        context = " ".join(context_parts)

        # Concatenamos pergunta e contexto para alimentar o modelo
        tokens = simple_tokenize(question + " " + context)
        idxs = [self.vocab.stoi.get(t, self.vocab.stoi["<unk>"]) for t in tokens]
        src = torch.tensor([idxs], dtype=torch.long, device=self.device)
        # Usamos a função de geração implementada na rede para obter
        # a sequência de resposta.
        pred = self.model.generate(src, self.vocab.bos_index, self.vocab.eos_index)
        # Convertemos os índices de volta para texto legível.
        return self.vocab.decode(pred)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executa o sistema de QA")
    parser.add_argument("question", nargs="?", help="Pergunta a ser respondida")
    parser.add_argument("--model_dir", default="spark_expert_model",
                        help="Diretório do modelo treinado")
    parser.add_argument("--index", default="spark_docs.index",
                        help="Caminho para o índice txtai")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Número de trechos recuperados")
    args = parser.parse_args()

    system = SparkQASystem(model_dir=args.model_dir, index_path=args.index,
                           top_k=args.top_k)

    if args.question:
        print(system.ask(args.question))
    else:
        # Modo demonstrativo com perguntas de exemplo
        for q in [
            "What is the spark sql cli set to??",
            "What is covered in the documentation?",
        ]:
            ans = system.ask(q)
            print(f"\n{q}\n-> {ans}")
