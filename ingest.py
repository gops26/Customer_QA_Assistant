import pandas as pd
from dotenv import load_dotenv
load_dotenv()

DATASET_FILE_PATH = "database/customers.csv"

EMBEDDING_MODEL = "text-embedding-3-small"

dataset = pd.read_csv(DATASET_FILE_PATH)

VECTOR_DB_PATH = "/vectordb"


print("**"*15)
print("creating chunks")

# Create a text chunk for each row
chunks = []
for _, row in dataset.iterrows():
    chunk = ", ".join(f"{col}: {row[col]}" for col in dataset.columns)
    chunks.append(chunk)

print(f"Total chunks created: {len(chunks)}")
print("Sample chunk:")

print(chunks[0])




from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import os

def create_embeddings(chunks:list[str]):
    embedding_dim = len(embeddings.embed_query("hello world"))

    if not os.makedirs(VECTOR_DB_PATH, exist_ok=True):
        index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore= InMemoryDocstore(),
            index_to_docstore_id={}

        )

        doc_ids = vector_store.add_texts(texts=chunks)
        print("="*20)
        print("Vector store created")
        print("="*20)

        vector_store.save_local(r"C:\Users\Gopinath\Desktop\Customer_QA_Assistant\vectordb", "bank_index")
        print("="*20)
        print("Vector store saved locally")
        print("="*20)


vector_store = FAISS.load_local(
    r"C:\Users\Gopinath\Desktop\Customer_QA_Assistant\vectordb",
    embeddings,
    allow_dangerous_deserialization=True
    )

def vector_search(query:str):
    return vector_store.similarity_search(query, k=1)

print(vector_search("shreya nair"))