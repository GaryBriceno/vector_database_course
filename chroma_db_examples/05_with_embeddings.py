from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

import chromadb
import os

from dotenv import load_dotenv
load_dotenv()

embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name="text-embedding-3-small",
)

client = chromadb.PersistentClient(path="platohedro.db")

collection = client.get_or_create_collection(
    name="docs_openai",
    embedding_function=embedding_fn
)

collection.add(
    ids=["1", "2"],
    documents=[
        "ChromaDB permite búsqueda semántica",
        "FastAPI es útil para exponer APIs de IA"
    ]
)

resultado = collection.query(
    query_texts=["vector database"],
    n_results=2,
    include=["documents", "metadatas", "embeddings"]
)

print(resultado)