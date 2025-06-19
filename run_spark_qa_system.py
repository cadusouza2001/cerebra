# run_spark_qa_system.py

from txtai.embeddings import Embeddings
from transformers import pipeline

class SparkQASystem:
    def __init__(self, index_path, model_path):
        print("Carregando o Bibliotecário (índice de embeddings)...")
        self.embeddings = Embeddings()
        self.embeddings.load(index_path)
        
        print("Carregando o Especialista (modelo de P&R ajustado)...")
        self.qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)
        print("Sistema está pronto.")

    def ask(self, question):
        print(f"\nRecebida a pergunta: {question}")
        
        # 1. Usa o Bibliotecário para encontrar o contexto relevante
        search_results = self.embeddings.search(question, limit=5)
        
        # Combina o texto dos principais resultados em um único contexto
        context = " ".join([result['text'] for result in search_results])
        
        print("--- Contexto relevante encontrado ---")
        
        # 2. Usa o Especialista para encontrar a resposta dentro do contexto
        result = self.qa_pipeline(question=question, context=context)
        
        return result

# --- USO ---
# Inicializa o sistema com os caminhos para seus componentes salvos
spark_expert = SparkQASystem(index_path="./spark_docs.index", model_path="./spark_expert_model")

# Faça suas perguntas!
answer1 = spark_expert.ask("Qual é o propósito da propriedade spark.driver.memory?")
print(f"Resposta Final: {answer1['answer']} (Confiança: {answer1['score']:.4f})")

answer2 = spark_expert.ask("Como posso armazenar em cache um Dataset?")
print(f"Resposta Final: {answer2['answer']} (Confiança: {answer2['score']:.4f})")

answer3 = spark_expert.ask("Que tipos de gerenciadores de cluster o Spark suporta?")
print(f"Resposta Final: {answer3['answer']} (Confiança: {answer3['score']:.4f})")