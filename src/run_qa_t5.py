import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class T5QASystem:
    """Sistema simples de QA usando modelo T5 fine-tunado."""

    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model.eval()

    def ask(self, question: str) -> str:
        inputs = self.tokenizer(question, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pergunte ao modelo T5")
    parser.add_argument("question", help="Pergunta a ser respondida")
    parser.add_argument("--model_dir", default="spark_t5_model", help="Diretório do modelo")
    args = parser.parse_args()

    system = T5QASystem(args.model_dir)
    print(system.ask(args.question))


if __name__ == "__main__":
    main()
