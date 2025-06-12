from langchain_community.tools import DuckDuckGoSearchResults
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from langgraph.graph.message import add_messages


load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# search_tool = DuckDuckGoSearchResults()
# result = search_tool.invoke('pakistan and bangladesh series result?')
# print(result)
llm = ChatAnthropic(model='claude-3-5-sonnet-20241022')

# tool calling
search_tool_tavily = TavilySearch(max_results=2)
# result = search_tool_tavily.invoke('pakistan and bangladesh series result?')
tools = [search_tool_tavily]
# print(result)
# print(tools)

graph_builder = StateGraph(State)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {'messages': [llm_with_tools.invoke(state['messages'])]}

graph_builder.add_node('chatbot', chatbot)

# tool_by_name = {tool.name: tool for tool in tools}
# print(tool_by_name)

# def toolnode(state:dict):
#     result = []
#     for tool_call in state['messages'][-1].tool_calls:
#         tool = tool_by_name[tool_call['name']]
#         print(tool)
#         observation= tool.invoke(tool_call['args'])
#         print(observation)

    #     result.append(ToolMessage(content=observation, tool_call_id=tool_call['id']))

    # return {'messages': result}

graph_builder.add_node('tools', ToolNode(tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge('tools', 'chatbot')
graph_builder.add_edge(START, 'chatbot')
graph = graph_builder.compile()

def graph_stream(user_input: str):
    for event in graph.stream({'messages': [{'role': 'user', 'content': user_input}]}):
        for value in event.values():
            print(f'Assistant: {value}')

while True:
    user = input('User: ')
    if user.lower() in ['q']:
        break

    graph_stream(user)
