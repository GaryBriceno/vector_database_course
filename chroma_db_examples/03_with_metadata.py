import chromadb

client = chromadb.PersistentClient(path="platohedro.db")

collection = client.get_or_create_collection(name="articulos")

collection.add(
    ids=["1", "2", "3"],
    documents=[
        "Aprender ChromaDB paso a paso",
        "Construir un buscador semantico de Python",
        "FastAPI para microservicios",
    ],
    metadatas=[
        {"tema": "vectordb", "nivel": "basico", "anio": 2000},
        {"tema": "python", "nivel": "basico", "anio": 2001},
        {"tema": "fastapi", "nivel": "intermedio", "anio": 2002},
    ]
)

resultado = collection.query(
    query_texts="Quiero aprender python",
    n_results=1
)

print(resultado)