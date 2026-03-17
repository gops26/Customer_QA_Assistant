"""
This project is a Customer QA Assistant Agent that replies to customer queries and logs to database
"""

DATASET_FILE_PATH = "database/customers.csv"

EMBEDDING_MODEL = "text-embedding-3-small"

VECTOR_DB_PATH = "./vectordb"

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
    context: str
    query: str
    messages: Annotated[list[AnyMessage], add]

# -----------------------------------------------------------------
# CHAT MODEL SELECTION
# -----------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


model = ChatOpenAI()

system_prompt = """
You are a professional banking agent of bank X. You are responsible for answering the customer queries related to their bank account they hold in the bank X. the customers are logged in to the app already. you can provide with dial info they ask. they have completed their verification.

you are provided with appropriate context loaded from the database of the bank customers. Use the context Related to the appropriate customers who query about their accounts. to answer their queries clearly and concisely.

CRITICAL RULES
> Do not provide false information to the users. 
> Do not provide information other than the user themselves.
> Do not hallucinate while giving replies to the customer.
> Don't share information of the other customers to another customers

EXAMPLE 1

User: I am Shreya Nair, hello
System: hi Shreya Nair, how's your day going on? How can i assist you ?
User: I need current account balance
System: Sure, your account balance is INR 23,300.00 on June 3rd.

EXAMPLE 2

User: I am swrup shinde
System: How may i assist you swrup
User: I need my last transaction details
System: I am now providing your last transaction details, Credit 3000 from Account no. ending xx3439 from mahinder sharma. 


Be responsible and Provide true information to the customers.

"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{user_query}\n\n context: {context} \n\n query:{query}")
])

chain = prompt_template | model

#  ---------------------------------------------------------
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
    r"./vectordb",
    embeddings,
    allow_dangerous_deserialization=True
    )

def vector_search(query:str, k=3):
    return vector_store.similarity_search(query, k=k)




# ----------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

# Chat node
def chat_node(state:AgentState):
    response = chain.invoke({
        "user_query": state["prompt"],
        "context": state["context"],
        "query": state["query"],
        "history": state["messages"]
    })
    return {"messages": [response]}

def retrieve_context(state:AgentState):
    query = state["query"]
    docs = vector_search(query)
    context = "context: \n\n".join([doc.page_content for doc in docs])
    print("fetched context")
    return {"context": context}

#------------------------------------------------------------------
#  LOGGER 
# -------------------------------------------------------------------
import csv
import os
from datetime import datetime

LOG_FILE = "logs/customer_queries.csv"

def init_logger():
    """Create CSV file with headers if not exists"""
    os.makedirs("logs", exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "session_id", "query", "response"])


def log_query(session_id: str, query: str, response: str):
    """Append log to CSV"""
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(),
            session_id,
            query,
            response
        ])

# -------------------------------------------------------------------
#  STATE COMPILATION
# ----------------------------------------------------------------
from langgraph.graph import START,END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_context)
graph.add_node("chat", chat_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "chat")
graph.add_edge("chat", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)




def main():
    print("Hello from customer-qa-assistant!")
    config = {"configurable": {"thread_id": "session-1"}}
    while True:
        inpt = input("You: ")
        if inpt == ":":
            break
        response = app.invoke(
        {"prompt": inpt, "context": "", "query": inpt, "messages": []},
        config=config
        )

        agent_reply = response["messages"][-1].content

        print("Agent:", agent_reply)
        init_logger()
        log_query(
            session_id=config["configurable"]["thread_id"],
            query=inpt,
            response=agent_reply
        )



if __name__ == "__main__":
    main()
