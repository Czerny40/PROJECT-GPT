from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
import streamlit as st

from utils.chat_utils import initialize_messages, send_message, load_chat_history
from utils.callback_utils import ChatCallbackHandler
from utils.document_utils import (
    format_documents,
    create_text_splitter,
    save_uploaded_file,
    create_cache_dir,
)
from utils.retriever_utils import create_ensemble_retriever

prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the following context and not your training data. If you don't know the answer just say you don't know. DON'T make anything up.
    
    Context: {context}
    Question:{question}
    """
)

llm = ChatOllama(
    model="jinbora/deepseek-r1-Bllossom:8b",
    temperature=0.2,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

initialize_messages()


@st.cache_resource(show_spinner="파일을 분석하고있어요...")
def embed_file(file):
    file_path = save_uploaded_file(file, "private_files")
    cache_dir = create_cache_dir(file.name, "private_embeddings")

    splitter = create_text_splitter()
    file_loader = UnstructuredFileLoader(file_path)
    docs = file_loader.load_and_split(text_splitter=splitter)

    embeddings = OllamaEmbeddings(model="jinbora/deepseek-r1-Bllossom:8b")

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        },
    )

    return create_ensemble_retriever(vector_retriever, docs)


st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📄",
)

st.title("PrivateGPT")

st.markdown(
    """
안녕하세요!

PrivateGPT는 업로드한 문서를 로컬 LLM 기반으로 질문에 답해드리는 서비스입니다!

먼저 사이드바에서 파일을 업로드해주세요!
"""
)

st.sidebar.markdown(
    """
### 지원하는 파일 형식
- PDF (.pdf)
- 텍스트 파일 (.txt)
- Word 문서 (.docx)

"""
)

with st.sidebar:
    file = st.file_uploader(
        ".txt, .pdf, .docx 파일을 업로드해주세요", type=["txt", "pdf", "docx"]
    )

if file:
    retriever = embed_file(file)
    send_message("무엇이든 물어보세요!", "ai", save=False)
    load_chat_history()
    message = st.chat_input("업로드한 문서에 대해 무엇이든 물어보세요")

    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever | RunnableLambda(format_documents),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []
