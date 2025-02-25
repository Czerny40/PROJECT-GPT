import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from utils.chat_utils import save_message


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message = ""  # 시작할 때 메시지 초기화
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        # 매번 message_box를 비우고 새로 표시
        self.message_box.markdown(self.message)
