from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore


def format_documents(docs):
    return "\n\n".join(document.page_content for document in docs)


def create_text_splitter(chunk_size=1000, chunk_overlap=200):
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def save_uploaded_file(file, directory):
    file_content = file.read()
    file_path = f"./.cache/{directory}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


def create_cache_dir(file_name, directory):
    return LocalFileStore(f"./.cache/{directory}/{file_name}")
