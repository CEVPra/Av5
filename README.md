# Sistema de Recuperação de Informações usando VectorDB e RetrievalQA
O presente projeto utiliza da biblioteca LangChain junto com FAISS para gerar um sistema de perguntas e respostas baseado em documentos, utilizando embeddings gerados pela API do OpenAI.

# Visão Geral
O sistema utiliza a API da OpenAI para gerar embeddings a partir de textos. Esses embeddings são indexados com FAISS, para permitir uma busca eficiente de respostas relevantes. O modelo de RetrievalQA é então usado para combinar a recuperação de documentos com a geração de respostas a partir do modelo da OpenAI.

# Tecnologias Utilizadas
1- FAISS
2- OpenAI
3- LangChain

# Instalar as dependências:
pip install -r requerimentos.txt
