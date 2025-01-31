import streamlit as st
from langchain.retrievers import WikipediaRetriever


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

with st.sidebar:
    choive = st.selectbox("사용할 항목을 골라주세요.", (
        "File",
        "Wikipedia"
        ),
    )
    if choive == "File":
        file = st.file_uploader(
            ".docx, .txt, .pdf 파일만 업로드해주세요.",
            type=["docx", "txt", "pdf"],
        )
    else:
        topic = st.text_input("검색할 키워드를 입력해주세요.")
        if topic:
            retriever = WikipediaRetriever()
            docs = retriever.get_relevant_documents(topic)
            st.write(docs)