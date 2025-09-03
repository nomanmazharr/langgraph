from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
import sqlite3

load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatbotState) -> ChatbotState:
    prompt = state['messages']
    response = model.invoke(prompt)

    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatbotState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)

def save_thread_name(thread_id: str, name: str):
    conn.execute("INSERT OR REPLACE INTO thread_names (thread_id, name) VALUES (?, ?)", (thread_id, name))
    conn.commit()

def retrieve_all_threads():
    threads_dict = {}
    # all_threads = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        thread_name = checkpoint.config.get("metadata", {}).get("thread_name", "New Chat")

        threads_dict[thread_id] = {
            "id": str(thread_id),
            "name": thread_name
        }

    return list(threads_dict.values())