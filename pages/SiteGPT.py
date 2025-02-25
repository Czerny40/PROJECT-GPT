from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import streamlit as st

from utils.chat_utils import initialize_messages, send_message, load_chat_history, save_message
from utils.callback_utils import ChatCallbackHandler

answer_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def get_answers(input):
    docs = input["docs"]
    question = input["question"]

    llm.streaming = False
    llm.callbacks = None
    answer_chain = answer_prompt | llm
    # answer = []
    # for doc in docs:
    #     result = answer_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answer.append(result.content)
    # st.write(answer)
    # return [answer_chain.invoke({"question": question, "context": doc.page_content}).content for doc in docs]
    return {
        "question": question,
        "answers": [
            {
                "answer": answer_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


def choose_answers(input):
    answers = input["answers"]
    question = input["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


# header와 footer를 soup에서 제외하는 함수
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ")


@st.cache_resource(show_spinner="Loading Website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=100,
    )

    loader = SitemapLoader(
        url,
        filter_urls=(
            [
                r"https:\/\/developers.cloudflare.com/ai-gateway.*",
                r"https:\/\/developers.cloudflare.com/vectorize.*",
                r"https:\/\/developers.cloudflare.com/workers-ai.*",
            ]
        ),
        parsing_function=parse_page,
    )
    # loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(
            model="text-embedding-3-small",
        ),
    )
    return vector_store.as_retriever()

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.2,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

initialize_messages()

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.title("SiteGPT")


st.markdown(
    """          
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL(.xml)",
        placeholder="https://example.com",
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Invalid URL. Please use a valid sitemap (.xml) URL.")
    else:
        send_message("웹사이트에 대해 궁금한 점을 물어보세요", "ai", save=False)
        load_chat_history()
        query = st.chat_input("웹사이트에 대해 궁금한 점을 물어보세요")

        retriever = load_website(url)


        if query:
            send_message(query, "human")
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answers)
            )
            with st.chat_message("ai"):
                result = chain.invoke(query)
                response = result.content.replace("$", "\$")
                st.markdown(response)

            save_message(response, "ai")
else:
    st.session_state["messages"] = []
