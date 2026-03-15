"""
This project is a Customer QA Assistant Agent that replies to customer queries and logs to database
"""
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
