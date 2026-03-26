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

