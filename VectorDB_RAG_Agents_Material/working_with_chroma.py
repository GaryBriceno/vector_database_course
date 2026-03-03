import shutil
import os

PERSIST_DIR = os.path.abspath("./VectorDB_RAG_Agents_Material/db")

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

print("DB cleaned")