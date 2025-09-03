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

try:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_names (
            thread_id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    conn.commit()
except sqlite3.OperationalError as e:
    print(f"Error creating thread_names table: {e}")
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatbotState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)

def save_thread_name(thread_id: str, name: str):
    try:
        conn.execute("INSERT OR REPLACE INTO thread_names (thread_id, name) VALUES (?, ?)", (thread_id, name))
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"Error saving the thread name: {e}")

def retrieve_all_threads():
    threads_dict = {}
    # all_threads = set()
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'")
        if cursor.fetchone():
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            for row in cursor.fetchall():
                thread_id = row[0]

                name_row = cursor.execute("SELECT name FROM thread_names where thread_id=?", (thread_id,)).fetchone()
                thread_name = name_row[0] if name_row else "New Chat"
                threads_dict[thread_id] = {
                    "id": str(thread_id),
                    "name": thread_name
                }
        return list(threads_dict.values())
    except sqlite3.OperationalError as e:
        print(f"Error retrieving threads: {e}")
        return []