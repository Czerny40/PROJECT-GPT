from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st


# AI 응답을 실시간으로 표시하기 위한 콜백 핸들러
class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        # AI 응답 시작 시 빈 메시지 박스 생성
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        # AI 응답 완료 시 메시지 저장
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        # 응답이 토큰로 생성될 때마다 실시간으로 화면에 표시
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model="deepseek-r1:7b",
    temperature=0.2,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

# 메시지 저장소 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 파일 임베딩 함수
@st.cache_resource(show_spinner="파일을 분석하고있어요...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n"],
        length_function=len,
    )

    file_loader = UnstructuredFileLoader(file_path)
    docs = file_loader.load_and_split(text_splitter=splitter)

    embeddings = OllamaEmbeddings(
        model="deepseek-r1:7b"
    )

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr", # MMR(Maximum Marginal Relevance) : 검색 결과의 품질과 다양성을 동시에 고려
        search_kwargs={
            "k": 4,  # 검색할 문서 수
            "fetch_k": 20,  # 후보 풀 크기
            "lambda_mult": 0.7,  # 다양성 vs 관련성 가중치 (1에 가까울수록 관련성 중시)
        },
    )

    # BM25 : 문서 검색을 위한 강력한 랭킹 알고리즘으로, 문서와 검색어 간의 관련성을 평가
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4  # 검색할 문서 수

    # 하이브리드 검색기 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],  # 각 검색기의 가중치
    )

    return ensemble_retriever

# 메시지 표시 함수
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

# 메시지 저장 함수
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

# 채팅 기록 불러오기 함수
def load_chat_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_documents(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the following context and not your training data. If you don't know the answer just say you don't know. DON'T make anything up.
    
    Context: {context}
    Question:{question}
    """
)

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
