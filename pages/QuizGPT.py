import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser


class JsonOutputParser(BaseOutputParser):

    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.2,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
    streaming=True,
)


def format_documents(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a helpful assistant that is role playing as a teacher.
            
        Based ONLY on the following context make 5 questions to test the user's knowledge about the text.
        
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
            
        Use (o) to signal the correct answer.

        Question examples:
            
        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
            
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
            
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model

        Your turn!
            
        Context: {context}
    """,
        )
    ]
)

questions_chain = {"context": format_documents} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    Your task is to:
    1. Format exam questions into JSON format
    2. Translate all content into native korean
    Answers with (o) are the correct ones.
     
   Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
     
    Example Output for Korean:
     
    ```json
    {{
        "questions": [
            {{
                "question": "바다의 색은 무엇인가요?",
                "answers": [
                    {{
                        "answer": "빨간색",
                        "correct": false
                    }},
                    {{
                        "answer": "노란색",
                        "correct": false
                    }},
                    {{
                        "answer": "초록색",
                        "correct": false
                    }},
                    {{
                        "answer": "파란색",
                        "correct": true
                    }}
                ]
            }}
        ]
    }}
    ```

    Important:
    - First convert the input to JSON format
    - Then translate all questions and answers to korean
    - Maintain the same structure and correct answer indicators
    
    Your turn!
    Questions: {context}
""",
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_resource(show_spinner="파일을 분석하고있어요...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n"],
        length_function=len,
    )

    file_loader = UnstructuredFileLoader(file_path)
    docs = file_loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_resource(show_spinner="퀴즈를 만들고있어요...")
def run_quiz_chain(_docs, topic):
    # questions_response = questions_chain.invoke(_docs)
    # formatting_response = formatting_chain.invoke(
    #     {"context": questions_response.content}
    # )
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_resource(show_spinner="퀴즈를 만들고있어요...")
def wiki_search(keyword):
    retriever = WikipediaRetriever(top_k_results=3, lang="ko")
    return retriever.get_relevant_documents(keyword)


def find_correct_answer(answers):
    for answer in answers:
        if answer["correct"]:
            return answer["answer"]
    return None


with st.sidebar:
    docs = None
    topic = None
    choive = st.selectbox(
        "사용할 항목을 골라주세요.",
        ("File", "Wikipedia"),
    )
    if choive == "File":
        file = st.file_uploader(
            ".docx, .txt, .pdf 파일만 업로드해주세요.",
            type=["docx", "txt", "pdf"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("검색할 키워드를 입력해주세요.")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
    안녕하세요!

    QuizGPT는 사용자의 학습을 위해 업로드한 문서 혹은 위키피디아를 바탕으로 GPT가 문제를 만들어주는 서비스입니다!

    먼저 사이드바에서 파일을 업로드, 또는 검색할 단어를 입력해주세요!
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    if response:
        with st.sidebar:
            switch = st.toggle("정답보기")

    with st.form("questions_form"):
        for idx, question in enumerate(response["questions"]):
            value = st.radio(
                f"Q{idx+1}. {question['question']}",
                [
                    f"{answer['answer']}"
                    for index, answer in enumerate(question["answers"])
                ],
                index=None,
            )
            isCorrect = False
            if value:
                isCorrect = {"answer": value[3:], "correct": True} in question[
                    "answers"
                ]
            if isCorrect:
                st.success("✅ 정답입니다!")
            elif value:
                if switch:
                    correct_answer = find_correct_answer(question["answers"])
                    st.error(f"❌ 오답입니다. (정답: {correct_answer})")
                else:
                    st.error("❌ 오답입니다.")
            st.divider()
        button = st.form_submit_button()
