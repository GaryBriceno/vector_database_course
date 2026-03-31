def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks

texto = """
ChromaDB es una base de datos vectorial que permite almacenar embeddings...
FastAPI permite construir APIs rápidas...
Los embeddings representan texto como vectores...
"""

chunks = chunk_text(texto, chunk_size=10, overlap=5)

for i, c in enumerate(chunks):
    print(f"Chunk {i}:\n{c}\n")


import chromadb

# 1) Cliente local persistente
client = chromadb.PersistentClient(path="platohedro.db")

# 2) Crear o recuperar colección
collection = client.get_or_create_collection(name="demo_chunking")

collection.add(
    ids=[f"doc1_chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    metadatas=[
        {
            "source": "doc1",
            "chunk_id": i
        }
        for i in range(len(chunks))
    ]
)

resultado = collection.get(
    include=["documents", "metadatas", "embeddings"]
)

from pprint import pprint
pprint(resultado)