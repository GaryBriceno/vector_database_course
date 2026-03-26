import chromadb

client = chromadb.PersistentClient(path="chroma_data")

collection = client.get_or_create_collection(name="articulos")

collection.add(
    ids=["1", "2", "3"],
    documents=[
        "Aprender ChromaDB paso a paso",
        "Construir un buscador semantico de Python",
        "FastAPI para microservicios",
    ],
    metadatas=[
        {"tema": "vectordb", "nivel": "basico"},
        {"tema": "python", "nivel": "basico"},
        {"tema": "fastapi", "nivel": "intermedio"},
    ]
)

resultado = collection.query(
    query_texts="Quiero aprender python",
    n_results=1
)

print(resultado)