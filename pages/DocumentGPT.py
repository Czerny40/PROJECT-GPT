from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
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

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        당신은 문서 분석 전문가입니다. 다음 규칙을 따르세요:
        1. 주어진 문서의 내용을 기반으로 답변하세요
        2. 제공된 정보가 있다면 반드시 그 내용을 포함하여 답변하세요.
        3. 모르는 내용은 솔직히 모른다고 하세요

        문서 내용:
        {context}
        """,
        ),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.2,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

initialize_messages()


@st.cache_resource(show_spinner="파일을 분석하고있어요...")
def embed_file(file):
    file_path = save_uploaded_file(file, "files")
    cache_dir = create_cache_dir(file.name, "embeddings")

    splitter = create_text_splitter()
    file_loader = UnstructuredFileLoader(file_path)
    docs = file_loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",  # MMR(Maximum Marginal Relevance) : 검색 결과의 품질과 다양성을 동시에 고려
        search_kwargs={
            "k": 4,  # 검색할 문서 수
            "fetch_k": 20,  # 후보 풀 크기
            "lambda_mult": 0.7,  # 다양성 vs 관련성 가중치 (1에 가까울수록 관련성 중시)
        },
    )

    return create_ensemble_retriever(vector_retriever, docs)


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)

st.title("DocumentGPT")

st.markdown(
    """
안녕하세요!

DocumentGPT는 업로드한 문서를 기반으로 질문에 답해드리는 서비스입니다!

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
