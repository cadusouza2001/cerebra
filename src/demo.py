"""Demonstration of the full QA system"""

from run_qa_system import SparkQASystem

system = SparkQASystem(index_path="spark_docs.index", model_path="spark_expert_model")

questions = [
    "What is Apache Spark?",
    "How to cache a Dataset?",
]

for q in questions:
    answer = system.ask(q)
    print(f"\n{q}\n-> {answer['answer']} (score {answer['score']:.4f})")
