import chromadb

client = chromadb.Client()

collection = client.get_or_create_collection(name="mi_primera_collection")

collection.add(
    ids=["1", "2", "3"],
    documents=[
        "Python es un lenguaje de programacion",
        "FastAPI sirve para construir APIs rapidas",
        "ChromaDB es una base de datos vectorial"
    ]
)

resultados = collection.query(
    query_texts="Para programar",
    n_results=1
)

print(resultados["ids"])
print(resultados["documents"])