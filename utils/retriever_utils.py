from langchain.retrievers import BM25Retriever, EnsembleRetriever


def create_ensemble_retriever(vector_retriever, docs, k=4, weights=[0.5, 0.5]):
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=weights
    )
    return ensemble_retriever
