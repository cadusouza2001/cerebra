"""Demonstração rápida do sistema de perguntas e respostas."""

from run_qa_system import SparkQASystem

system = SparkQASystem(model_dir="spark_expert_model")

questions = [
    "What is Apache Spark?",
    "How to cache a Dataset?",
]

for q in questions:
    answer = system.ask(q)
    print(f"\n{q}\n-> {answer}")
