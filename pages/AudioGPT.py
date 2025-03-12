import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
import yt_dlp
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.storage import LocalFileStore
from operator import itemgetter


from utils.document_utils import create_text_splitter
from utils.chat_utils import (
    initialize_messages,
    send_message,
    load_chat_history,
    save_message,
    save_memory,
)
from utils.callback_utils import ChatCallbackHandler

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.1,
)

chat_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

initialize_messages()
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        llm=llm, max_token_limit=1000, return_messages=True
    )
if "is_summary" not in st.session_state:
    st.session_state["is_summary"] = False


@st.cache_data()
def extract_audio(video_path):
    audio_path = video_path.replace("mp4", "mp3")
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    os.makedirs(chunks_folder, exist_ok=True)

    track = AudioSegment.from_mp3(audio_path)
    overlap_size = 10 * 1000
    chunk_len = chunk_size * 60 * 1000 - overlap_size
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len + overlap_size

        chunk = track[start_time:end_time]

        chunk.export(f"{chunks_folder}/chunk_{i:02d}.mp3", format="mp3")


@st.cache_data()
def transcript_chunks(chunks_folder, destination):
    chunks = glob.glob(f"{chunks_folder}/*.mp3")
    chunks.sort()
    for chunk in chunks:
        with open(chunk, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", file=audio_file, language="en"
            )
            text_file.write(transcript.text)


@st.cache_data()
def download_audio_from_youtube(youtube_url, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


@st.cache_data()
def summary_transcript(transcript_path):
    loader = TextLoader(transcript_path)
    splitter = create_text_splitter(chunk_size=800, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)

    first_summary_prompt = ChatPromptTemplate.from_template(
        """
            Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:  
    """
    )

    first_summary_chain = first_summary_prompt | llm | StrOutputParser()

    summary = first_summary_chain.invoke({"text": docs[0].page_content})

    refine_prompt = ChatPromptTemplate.from_template(
        """
        Your job is to produce a final summary.
        We have provided an existing summary up to a certain point: {existing_summary}
        We have the opportunity to refine the existing summary (only if needed) with some more context below.
        ------------
        {context}
        ------------
        Given the new context, refine the original summary.
        If the context isn't useful, RETURN the original summary.
        """
    )

    refine_chain = refine_prompt | llm | StrOutputParser()

    for i, doc in enumerate(docs[1:]):
        refined_summary = refine_chain.invoke(
            {"existing_summary": summary, "context": doc.page_content}
        )
        summary = refined_summary

    return summary, len(docs) - 1


@st.cache_resource()
def embed_file(file_path):

    cache_dir = LocalFileStore(f"./.cache/embeddings/{os.path.basename(file_path)}")
    loader = TextLoader(file_path)
    splitter = create_text_splitter(chunk_size=800, chunk_overlap=100)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


st.set_page_config(
    page_title="AudioGPT",
    page_icon="💬",
)

st.markdown(
    """
안녕하세요!

AudioGPT는 업로드한 영상 파일 혹은 유튜브 링크를 기반으로 질문에 답해드리는 서비스입니다!

먼저 사이드바에서 파일 혹은 링크를 제공해주세요!
"""
)

with st.sidebar:

    choice = st.selectbox(
        "사용할 항목을 골라주세요.",
        ("File", "Youtube"),
    )

    audio_path = None

    if choice == "File":
        video = st.file_uploader(
            "Video",
            type=["mp4", "avi", "mkv", "mov"],
        )

        if video:
            with st.spinner("파일을 불러오는 중입니다..."):
                video_content = video.read()
                video_path = f"./.cache/{video.name}"
                with open(video_path, "wb") as f:
                    f.write(video_content)
                extract_audio(video_path)
                audio_path = video_path.replace("mp4", "mp3")

    elif choice == "Youtube":
        youtube_url = st.text_input("유튜브 URL을 입력해주세요")

        if youtube_url:
            if st.button("오디오 추출"):
                with st.spinner("유튜브에서 오디오를 추출 중입니다..."):
                    output_path = "./.cache/youtube_audio"
                    download_audio_from_youtube(youtube_url, output_path)
                    audio_path = f"{output_path}.mp3"
                    st.success("오디오 추출이 완료되었습니다!")

    if audio_path and os.path.exists(audio_path):
        st.audio(audio_path)

        chunks_folder = "./.cache/chunks"
        if audio_path.endswith(".mp3"):
            transcript_path = audio_path.replace(".mp3", ".txt")
        else:
            transcript_path = f"{audio_path}.txt"

        with st.spinner("오디오 파일을 분할중입니다..."):
            cut_audio_in_chunks(audio_path, 10, chunks_folder)

        transcript_start = st.button("대본추출")
        if transcript_start:
            with st.spinner("텍스트로 변환 중입니다..."):
                transcript_chunks(chunks_folder, transcript_path)

if audio_path and os.path.exists(audio_path):
    transcript_tab, summary_tab, qa_tab = st.tabs(["대본", "요약", "질문"])

    with transcript_tab:
        if os.path.exists(transcript_path):
            with open(transcript_path, "r") as file:
                st.write(file.read())
        else:
            st.warning(
                "대본 파일이 존재하지 않습니다. '대본추출' 버튼을 클릭하여 대본을 생성해주세요."
            )

    with summary_tab:
        summary_start = st.button("요약하기")

        if summary_start or st.session_state["is_summary"]:
            if os.path.exists(transcript_path):
                progress_bar = st.progress(0)
                status_text = st.empty()

                with st.status("요약중입니다...") as status:
                    status.update(label="요약 준비중...")
                    summary, total_docs = summary_transcript(transcript_path)

                    progress_bar.progress(100, text=f"총 {total_docs}개 문서 처리 완료")

                    status.update(label=f"요약이 완료되었습니다!", state="complete")

                st.write(summary)
                st.session_state["is_summary"] = True
            else:
                st.warning(
                    "대본 파일이 존재하지 않습니다. '대본추출' 버튼을 클릭하여 대본을 생성해주세요."
                )

    with qa_tab:
        if os.path.exists(transcript_path):
            retriever = embed_file(transcript_path)
            query = st.chat_input("영상에서 궁금한 내용을 물어보세요.")

            load_chat_history()

            chat_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an AI assistant that answers questions about video content.
                        Use both the information provided in the video transcript and our previous conversation history to answer the user's questions accurately.
                        
                        Follow these rules:
                        1. Prioritize information from the provided video transcript.
                        2. Remember and reference our previous conversation to provide contextually relevant answers.
                        3. For questions not related to the video content, respond as in a normal conversation.
                        4. Provide concise and clear answers.
                        5. Explain in easy-to-understand language.
                        6. Must answer in Korean.
                        
                        Here is the relevant content from the video transcript:
                        {context}
                        
                        Previous conversation history:
                        {history}
                        """,
                    ),
                    ("human", "{question}"),
                ]
            )
            if query:
                send_message(query, "human")
                chain = (
                    {
                        "context": retriever,
                        "question": RunnablePassthrough(),
                        "history": RunnableLambda(
                            st.session_state.memory.load_memory_variables
                        )
                        | itemgetter("history"),
                    }
                    | chat_prompt
                    | chat_llm
                )
                with st.chat_message("ai"):
                    result = chain.invoke(query)
                    save_memory(query, result.content)
        else:
            st.warning(
                "대본 파일이 존재하지 않습니다. '대본추출' 버튼을 클릭하여 대본을 생성해주세요."
            )
else:
    st.info("사이드바에서 영상 파일을 업로드하거나 유튜브 링크를 입력해주세요.")

    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationBufferMemory(
        llm=llm, max_token_limit=1000, return_messages=True
    )
    st.session_state["is_summary"] = False
