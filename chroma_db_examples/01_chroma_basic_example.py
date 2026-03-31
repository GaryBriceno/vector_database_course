import chromadb

client = chromadb.Client()

collection = client.get_or_create_collection(name="platohedro")

collection.add(
    ids=["1", "2", "3"],
    documents=[
        "Python es un lenguaje de programacion de scripts",
        "FastAPI sirve para construir APIs rapidas y es un framework",
        "ChromaDB es una base de datos vectorial para embeddings"
    ]
)

resultados = collection.query(
    query_texts="Voy a programar",
    n_results=1
)

print(resultados["ids"])
print(resultados["documents"])