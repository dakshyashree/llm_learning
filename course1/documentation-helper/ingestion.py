import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# 768-dim MPNet embeddings
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")

def ingest_docs():
    # 1) Point exactly at your local "_build/html" folder (absolute path!)
    loader = ReadTheDocsLoader(
        r"\Users\daksh\Desktop\Dakshu\Projects\llm_learning\course1\documentation-helper\langchain-docs"
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} raw pages")

    # 2) Split into ~600-char chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    # 3) (Optional) Fix up metadata["source"] if needed
    for doc in documents:
        original = doc.metadata.get("source", "")
        # If source came back as "langchain-docs/…", ensure it starts with "https://"
        if original.startswith("langchain-docs"):
            doc.metadata["source"] = "https://" + original
        # Otherwise leave it as is (e.g. it might already be a proper file:// or https:// URL)

    print(f"Going to add {len(documents)} chunks to Pinecone…")

    # 4) Upsert into Pinecone
    PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name="langchain-doc-index"
    )
    print("*** Loading to vectorstore done ***")


def ingest_docs2()-> None:
    from langchain_community.document_loaders import FireCrawlLoader
    langchain_documents_base_urls = [
        "https://python.langchain.com/docs/integrations/chat//",
        "https://python.langchain.com/docs/integrations/llms/",
        "https://python.langchain.com/docs/integrations/text_embedding/",
        "https://python.langchain.com/docs/integrations/document_loaders/",
        "https://python.langchain.com/docs/integrations/document_transformers/",
        "https://python.langchain.com/docs/integrations/vectorstores/",
        "https://python.langchain.com/docs/integrations/retrievers/",
        "https://python.langchain.com/docs/integrations/tools/",
        "https://python.langchain.com/docs/integrations/stores/",
        "https://python.langchain.com/docs/integrations/llm_caching/",
        "https://python.langchain.com/docs/integrations/graphs/",
        "https://python.langchain.com/docs/integrations/memory/",
        "https://python.langchain.com/docs/integrations/callbacks/",
        "https://python.langchain.com/docs/integrations/chat_loaders/",
        "https://python.langchain.com/docs/concepts/",
    ]

    langchain_documents_base_urls2 = [
        "https://python.langchain.com/docs/integrations/chat/"
    ]
    for url in langchain_documents_base_urls2:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="scrape",
        )
        docs = loader.load()

        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name="firecrawl-index"
        )
        print(f"****Loading {url}* to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()