"""Demonstração rápida do sistema de perguntas e respostas."""

# Pequeno script utilitário para testar o modelo já treinado.
# Ele ilustra a fase de inferência: passamos uma pergunta e
# recebemos a resposta gerada pela rede neural.

from run_qa_system import SparkQASystem

system = SparkQASystem(model_dir="spark_expert_model")

# Lista de perguntas que usaremos como exemplo
questions = [
    "What is Apache Spark?",
    "How to cache a Dataset?",
]

for q in questions:
    answer = system.ask(q)
    print(f"\n{q}\n-> {answer}")
