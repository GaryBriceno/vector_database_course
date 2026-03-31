import chromadb

client = chromadb.PersistentClient(path="platohedro.db")

collection = client.get_or_create_collection(name="notas")

collection.add(
    ids=["1", "2"],
    documents=[
        "Aprender ChromaDB paso a paso",
        "Construir un buscador semantico de Python"
    ]
)

resultado = collection.query(
    query_texts="Quiero aprender busqueda semantica",
    n_results=1
)

print(resultado)