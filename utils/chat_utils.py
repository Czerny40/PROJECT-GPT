import streamlit as st

def initialize_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def load_chat_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def save_memory(input, output):
    st.session_state["memory"].save_context({"input": input}, {"output": output})