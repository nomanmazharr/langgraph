import streamlit as st
from chatbot_tool_backend import workflow, retrieve_all_threads, save_thread_name, model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import json

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id, name="New Chat"):
    if not any(t['id'] == thread_id for t in st.session_state['chat_threads']):
        st.session_state['chat_threads'].append({"id": thread_id, "name": name})

def generate_thread_title(user_input, workflow):
    # Use the workflow instead of calling model directly
    prompt = f"""
    Summarize the following message into a short, clear title, understand semantic meaning if it's a greeting say it's greeting like that. (max 3 words).
    Message: "{user_input}"
    """
    response = model.invoke([HumanMessage(content=prompt)])
    return response.content

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state["message_history"] = []

def load_conversation(thread_id):
    state = workflow.get_state(config = {'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()


st.sidebar.title('Langgraph Bot')
if st.sidebar.button('New Chat'):
    reset_chat()


st.sidebar.header('My Conversations')

if st.session_state['chat_threads']:
    for thread in st.session_state['chat_threads'][::-1]:
        if st.sidebar.button(thread['name'], key=thread['id']):
            st.session_state['thread_id'] = thread['id']
            messages = load_conversation(thread['id'])

            temp_messages = []

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = 'user'
                else:
                    role = 'assistant'

                temp_messages.append({'role': role, 'content': msg.content})
            
            st.session_state['message_history'] = temp_messages


for messages in st.session_state['message_history']:
    with st.chat_message(messages['role']):
        st.text(messages['content'])

user_input = st.chat_input('Type here')


if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    thread_id = st.session_state['thread_id']
    t = next((thr for thr in st.session_state["chat_threads"] if thr['id'] == thread_id), None)
    current_name = "New Chat"
    if t:
        current_name = t['name']

    if t is None:
        current_name = generate_thread_title(user_input, workflow)
        save_thread_name(thread_id, current_name)
        add_thread(thread_id, current_name)
    
    CONFIG = {"configurable": {'thread_id': thread_id},
              "run_name": "chat_turn"}
    
    with st.chat_message('assistant'):
        
        status_holder = {'box': None}
        
        result = workflow.invoke(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG
            )
        messages = result.get('messages', [])
        
        ai_message = ""
        for msg in messages[-2:]:  # Check last two messages for ToolMessage and AIMessage
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "tool")
                try:
                    content = json.loads(msg.content) if msg.content else {"error": "No tool output"}
                except json.JSONDecodeError:
                    content = {"result": msg.content}
                with st.status(f"Using tool {tool_name} ...", expanded=True) as box:
                    status_holder["box"] = box
                    st.json(content)
            elif isinstance(msg, AIMessage) and msg.content:
                ai_message = msg.content
        
        if not ai_message:
            ai_message = "No response generated."
        
        st.write(ai_message)
        
        if status_holder['box'] is not None:
            status_holder['box'].update(
                label="Tool Finished", state="complete", expanded=False
            )
    st.session_state['message_history'].append({'role': 'assistant', 
                                                'content': ai_message})
    
    if t is None:
        st.rerun()