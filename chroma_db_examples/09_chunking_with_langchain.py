from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=15,
    chunk_overlap=10
)

texto = """
ChromaDB es una base de datos vectorial que permite almacenar embeddings...
FastAPI permite construir APIs rápidas...
Los embeddings representan texto como vectores...
"""

chunks = splitter.split_text(texto)


for i, c in enumerate(chunks):
    print(f"Chunk {i}:\n{c}\n")

