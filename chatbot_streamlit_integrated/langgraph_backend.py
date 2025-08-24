from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatbotState) -> ChatbotState:
    prompt = state['messages']
    response = model.invoke(prompt)

    return {"messages": [response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatbotState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)