import faiss
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

texts = [
   "Aquecimento global é nome dado ao fenômeno de aumento anormal das temperaturas do planeta Terra, tomando como referência de medição os níveis pré-industriais. Esse aumento é causado pela emissão de gases do efeito estufa (GEE), como o dióxido de carbono, proveniente da atividade antrópica e das alterações que os seres humanos provocam no meio ambiente."
]

def create_embeddings(texts):
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    return embeddings.embed_documents(texts)

def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings).astype(np.float32))
    return faiss_index

def setup_retrieval_qa(faiss_index, texts):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS(faiss_index, embeddings)
    llm = OpenAI(temperature=0)
    qa = RetrievalQA(combine_docs_chain=llm, retriever=vector_store.as_retriever())
    return qa

def main():
    embeddings = create_embeddings(texts)
    faiss_index = create_faiss_index(embeddings)
    qa = setup_retrieval_qa(faiss_index, texts)

    while True:
        question = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
        if question.lower() == 'sair':
            print("Encerrando o sistema.")
            break
        response = qa.run(question)
        print(f"Resposta: {response}")

if __name__ == "__main__":
    main()