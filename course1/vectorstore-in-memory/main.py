import os

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS

if __name__ == "__main__":
    print("hello")
    pdf_path = "/Users/daksh/Desktop/Dakshu/Projects/llm_learning/course1/vectorstore-in-memory/2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=document)

    embedding = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embedding, allow_dangerous_deserialization=True
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        ChatOllama(model="deepseek-R1"), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])
