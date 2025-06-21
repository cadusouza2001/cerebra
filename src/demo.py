"""Demonstration of the full QA system"""

# Pequeno script de demonstração. Instancia o sistema completo e faz
# algumas perguntas para mostrar o fluxo: recuperação de contexto +
# geração da resposta.

from run_qa_system import SparkQASystem

system = SparkQASystem(index_path="spark_docs.index", model_path="spark_expert_model")

questions = [
    "What is Apache Spark?",
    "How to cache a Dataset?",
]

# Executamos o método ask() para cada pergunta e exibimos a resposta
# junto com a pontuação de confiança.

for q in questions:
    # O método ask combina busca + geração para responder.
    answer = system.ask(q)
    print(f"\n{q}\n-> {answer['answer']} (score {answer['score']:.4f})")
