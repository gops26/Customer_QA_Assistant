"""
This project is a Customer QA Assistant Agent that replies to customer queries and logs to database
"""

DATASET_FILE_PATH = "database/customers.csv"

EMBEDDING_MODEL = "text-embedding-3-small"

VECTOR_DB_PATH = "/vectordb"

LLM_MODEL="" #gpt-3.5-turbo 

# ---------------------------------------------------------------
# STATE Definition
# -----------------------------------------------------------------
from typing_extensions import TypedDict
from typing import Annotated
from langchain.messages import AnyMessage
from operator import add
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    prompt: str
    messages: Annotated[list[AnyMessage], add]

# -----------------------------------------------------------------
# CHAT MODEL SELECTION
# -----------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage

from langchain_core.prompts import ChatPromptTemplate 


model = ChatOpenAI()

# prompt = ChatPromptTemplate.from_messages(
#     ("system", "")
# )

# ----------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

def chat_node(state:AgentState):
    return {
        "messages":[
            model.invoke(
                [
                    SystemMessage(content="you are an helpful assistant")
                ]
                + state["messages"]

                + [HumanMessage(content=state["prompt"])]
            )
        ]
    }
# ---------------------------------------------------------
# DATASET PROCESSING & EMBEDDING/VECTOR STORE CREATION
# ---------------------------------------------------------
import pandas as pd



dataset = pd.read_csv(DATASET_FILE_PATH)

def create_chunks():
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
    return chunks



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

        vector_store.save_local(r"\vectordb", "index")
        print("="*20)
        print("Vector store saved locally")
        print("="*20)


vector_store = FAISS.load_local(
    r"/vectordb",
    embeddings,
    allow_dangerous_deserialization=True
    )

def vector_search(query:str):
    return vector_store.similarity_search(query, k=1)


# -------------------------------------------------------------------
#  STATE COMPILATION
# ----------------------------------------------------------------
from langgraph.graph import START,END, StateGraph

graph = StateGraph(AgentState)

graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

app = graph.compile()




def main():
    print("Hello from customer-qa-assistant!")


    while True:
        inpt = input()
        if inpt == ":":
            break
        response = app.invoke({"prompt":inpt,"messages":[]})
        print(response["messages"][-1].content)



if __name__ == "__main__":
    main()
