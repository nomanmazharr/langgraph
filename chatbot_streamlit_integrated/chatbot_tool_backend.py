from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import requests
from langchain_core.messages import BaseMessage
import sqlite3
import os

load_dotenv()
stock_api = os.getenv("STOCK_API_KEY")

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

# Tools 
search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result, "tool": "Calculator V-1.0"}
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_api}"
    r = requests.get(url)
    return r.json()


tools = [search_tool, get_stock_price, calculator]
llm_with_tools = model.bind_tools(tools)


class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatbotState) -> ChatbotState:
    prompt = state['messages']
    response = llm_with_tools.invoke(prompt)

    return {"messages": [response]}

tool_node = ToolNode(tools)


conn = sqlite3.connect(database='chatbot_tool.db', check_same_thread=False)

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
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
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