from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from IPython.display import Image, display


load_dotenv()
class Inputtext(TypedDict):
    messages: Annotated[list, add_messages]
    # messages: list


graph_builder = StateGraph(Inputtext)
llm = ChatAnthropic(model_name='claude-3-5-sonnet-20241022', streaming=True)

def chatbot(state: Inputtext):
    return {'messages': llm.invoke(state['messages'])}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, 'chatbot')
graph = graph_builder.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))

def stream_graph(user_input: str):
    for event in graph.stream({'messages': [{'role': 'user', 'content': user_input}]}):
        print(f'Full event: {event}')
        for value in event.values():
            print(f'Assistant: {value['messages'].content}')
            # print(value)

while True:
    user = input('User: ')
    if user.lower() in ['q', 'quit', 'exit']:
        break
    stream_graph(user)
# print(graph.stream)
