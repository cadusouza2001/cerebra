# run_spark_qa_system.py

from txtai.embeddings import Embeddings
from transformers import pipeline

# Componente de inferência do projeto. Aqui juntamos o "Bibliotecário"
# (índice de embeddings) e o "Especialista" (modelo ajustado) para
# responder perguntas novas.

class SparkQASystem:
    def __init__(self, index_path, model_path):
        print("Carregando o Bibliotecário (índice de embeddings)...")
        self.embeddings = Embeddings()
        self.embeddings.load(index_path)

        print("Carregando o Especialista (modelo de P&R ajustado)...")
        # A pipeline da HuggingFace simplifica todo o processo de tokenização
        # e geração da resposta usando o modelo treinado em train.py.
        self.qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)
        print("Sistema está pronto.")

    def ask(self, question):
        print(f"\nRecebida a pergunta: {question}")

        # 1. Usa o Bibliotecário para encontrar o contexto relevante
        # Aqui realizamos a etapa de "retrieval" da arquitetura RAG:
        # a busca é feita nos embeddings para encontrar os trechos mais
        # parecidos semanticamente com a pergunta.
        search_results = self.embeddings.search(question, limit=5)

        # Combina o texto dos principais resultados em um único contexto
        # Concatenamos os textos recuperados para formar o contexto de entrada
        # do modelo de QA.
        context = " ".join([result['text'] for result in search_results])

        print("--- Contexto relevante encontrado ---")

        # 2. Usa o Especialista para encontrar a resposta dentro do contexto
        # A pipeline executa o modelo seq2seq/transformer que aprendemos na
        # disciplina, gerando a resposta condicionada ao contexto recuperado.
        result = self.qa_pipeline(question=question, context=context)

        return result

# --- USO ---
# Exemplo simples de uso em que inicializamos o sistema com o índice e
# o modelo treinado. A partir daqui podemos fazer perguntas de forma
# interativa ou integrar a outro aplicativo.
spark_expert = SparkQASystem(index_path="./spark_docs.index", model_path="./spark_expert_model")

# Faça suas perguntas!
answer1 = spark_expert.ask("Qual é o propósito da propriedade spark.driver.memory?")
print(f"Resposta Final: {answer1['answer']} (Confiança: {answer1['score']:.4f})")

answer2 = spark_expert.ask("Como posso armazenar em cache um Dataset?")
print(f"Resposta Final: {answer2['answer']} (Confiança: {answer2['score']:.4f})")

answer3 = spark_expert.ask("Que tipos de gerenciadores de cluster o Spark suporta?")
print(f"Resposta Final: {answer3['answer']} (Confiança: {answer3['score']:.4f})")