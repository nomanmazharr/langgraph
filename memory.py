# from langgraph.checkpoint.memory import MemorySaver
# from langchain_anthropic import ChatAnthropic
# from langgraph.graph import StateGraph, START
# from typing_extensions import TypedDict
# from langgraph.graph.message import add_messages
# from typing import Annotated
# from langchain_tavily import TavilySearch
# from dotenv import load_dotenv
# from langgraph.prebuilt import ToolNode, tools_condition

# load_dotenv()

# class State(TypedDict):
#     messages: Annotated[list, add_messages]




# graph_builder = StateGraph(State)

# # tool = TavilySearch(max_results=2)
# # tools = [tool]


# llm = ChatAnthropic(model='claude-3-5-sonnet-20241022')
# # llm_with_tools = llm.bind_tools(tools)


# def chatbot(state: State):
#     return {'messages': [llm.invoke(state['messages'])]}

# graph_builder.add_node('chatbot', chatbot)
# # tool_node = ToolNode(tools=tools)
# # graph_builder.add_node('tools', tool_node)

# # graph_builder.add_conditional_edges(
# #     'chatbot',
# #     tools_condition
# # )

# # graph_builder.add_edge('tools', 'chatbot')
# graph_builder.set_entry_point('chatbot')



# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)



# config = {"configurable": {"thread_id": "1"}}

# # def graph_stream(user_input: str):
# #     for event in graph.stream({'messages': [{'role': 'user', 'content': user_input}]}):
# #         for value in event.values():
# #             print(f'Assistant: {value}')
# user = 'multiply the numbers'
# events = graph.stream(
#     {'messages': [{'role': 'user', 'content':user}]},
#     config,
#     stream_mode="values"
# )

# for event in events:
#     result = event['messages'][-1]
#     print(result)
# # while True:
# #     user = input('User: ')
# #     if user.lower() in ['q']:
# #         break

# #     graph_stream(user)

from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated
from dotenv import load_dotenv

load_dotenv()

# Define state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build graph
graph_builder = StateGraph(State)

# Initialize LLM
llm = ChatAnthropic(model='claude-3-5-sonnet-20241022')

# Define chatbot node logic
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Add node and entry point
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")

# Compile graph with memory saver
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Config for thread
config = {"configurable": {"thread_id": "1"}}


# 🧠 Interactive Chat Loop
print("Start chatting with the assistant. Type 'exit' or 'quit' to end.")
while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ("exit", "quit"):
        print("Ending conversation. Goodbye!")
        break

    # Stream user message through the graph
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values"
    )

    # Print latest message from assistant
    for event in events:
        latest_message = event["messages"]
        # print(f"Assistant: {latest_message.content}")
        print('Assistant:', latest_message)