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
        
        As a final step, translate the questions and answers you create into native Korean.

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
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
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


with st.sidebar:
    docs = None
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
            retriever = WikipediaRetriever(top_k_results=3, lang="ko")
            with st.status("검색중이에요 잠시만 기다려주세요."):
                docs = retriever.get_relevant_documents(topic)

if not docs:
    st.markdown(
        """
    안녕하세요!

    QuizGPT는 사용자의 학습을 위해 업로드한 문서 혹은 위키피디아를 바탕으로 GPT가 문제를 만들어주는 서비스입니다!

    먼저 사이드바에서 파일을 업로드, 또는 검색할 단어를 입력해주세요!
    """
    )
else:

    start = st.button("퀴즈 만들기")

    if start:
        with st.status("퀴즈를 만들고 있어요..."):
            # questions_response = questions_chain.invoke(docs)
            # formatting_response = formatting_chain.invoke(
            #     {"context": questions_response.content}
            # )
            chain = {"context": questions_chain} | formatting_chain | output_parser
            response = chain.invoke(docs)
            st.write(response)
