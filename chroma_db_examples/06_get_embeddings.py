import chromadb

# 1) Cliente local persistente
client = chromadb.PersistentClient(path="./chroma_data")

# 2) Crear o recuperar colección
collection = client.get_or_create_collection(name="demo_embeddings")

# 3) Agregar un registro
collection.add(
    ids=["doc_1"],
    documents=["ChromaDB es una base de datos vectorial para búsqueda semántica"],
    metadatas=[{"tema": "vectordb", "nivel": "basico"}]
)

# 4) Recuperar el registro incluyendo embeddings
resultado = collection.get(
    ids=["doc_1"],
    include=["documents", "metadatas", "embeddings"]
)

print("REGISTRO COMPLETO:")
print(resultado)

embedding = resultado["embeddings"][0]

print("\nTIPO:", type(embedding))
print("DIMENSIONES DEL VECTOR:", len(embedding))
print("PRIMERAS 10 DIMENSIONES:")
print(embedding[:10])