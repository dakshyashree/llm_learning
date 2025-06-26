# take a medium article, load it, split it up to chunks,embed everything, store everything
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Ingestion")

    # using openai embedding here

    loader = TextLoader(
        r"C:\Users\daksh\Desktop\Dakshu\Projects\llm_learning\course1\intro-to-vector-dbs\mediumblog1.txt",
        encoding="utf-8",
    )
    document = loader.load()

    print("splitting...")
    text_spitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_spitter.split_documents(document)
    print(f"created{len(texts)} chunks")

    # embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("ingesting")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
