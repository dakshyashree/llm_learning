import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("Retrieving")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatOllama(model="deepseek-R1")
    query = "What is Pinecone in Machine Learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    #result = chain.invoke(input={})
    #print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    """create_stuff_documents_chain

    This helper (from langchain.chains.combine_documents.stuff) builds a chain that simply “stuffs” all retrieved document chunks into a single prompt and sends it to the LLM.
    In other words, instead of splitting the answer process into map/reduce steps, it concatenates (stuffs) every chunk into one context block.
    combine_docs_chain defines the logic for “given a set of documents and a question, turn them into one prompt and call the LLM.”"""
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    """That line is wiring together two pieces into a single “question → answer” pipeline:
    vectorstore.as_retriever()
    Converts your Pinecone‐backed vectorstore into a Retriever object.
    Under the hood, calling as_retriever() means “when I give you a query, go ask Pinecone for the top‐K most similar chunks, and return them as a list of Document objects.”

    combine_docs_chain
    This is the chain you built with create_stuff_documents_chain(...) (or a similar helper). It knows how to take a list of retrieved chunks, plug them into a prompt template (e.g. with {context} + {question}), and send that to your LLM to generate a final answer.

    create_retrieval_chain(...)
    What you’re doing here is building a “retrieval‐augmented” chain that reads:
    “Given a user question, first use vectorstore.as_retriever() to fetch the most relevant chunks from Pinecone. Then feed those chunks into combine_docs_chain to have the LLM stitch them together and produce a coherent answer.”"""

    result = retrieval_chain.invoke(input={"input":query})

    print(result)

    template = """Use the following pieces of context to answer the question at the end.
    If you dont know the answer, just say that you dont know, dont try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.
    
    {context}
    
    Question: {question}
    
    Helpful answer:
    """
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )
    """
    This line is using LangChain’s new “Runnable” API (introduced in v0.2) to build a small RAG pipeline in one expression. In plain terms, it means:

    Take a user’s question (“question”) and send it straight through
    
    "question": RunnablePassthrough()
    RunnablePassthrough() is just a no-op that makes the raw question string available downstream under the name "question".
    
    Fed that same question into your vector store retriever to fetch context
    
    "context": vectorstore.as_retriever() | format_docs
    vectorstore.as_retriever() is a retriever runnable that, when given the question, returns a list of matching Document objects from Pinecone (or whatever vector store).
    
    Piped (|) into format_docs, which is another runnable that takes those Document objects and renders them into a single chunk of text (e.g. concatenating titles, snippets, etc.) so you have one big “context” string.
    
    In other words:
    
    question  ──▶ retriever ──▶ [Document1, Document2, …] ──▶ format_docs ──▶ "Context as one text block"
    and that final text block is made available under the "context" key.
    
    Now you have two pieces ready: {"context": <formatted-docs>, "question": <raw-question>}
    Those get fed into the next runnable, custom_rag_prompt, which is typically a PromptTemplate (wrapped as a Runnable) that expects exactly two inputs—{context} and {question}—and spits out a single prompt string, for example:
    
    “Here’s everything we know:\n{context}\n\nNow answer:\n{question}”
    Finally, pipe that filled-in prompt into your LLM

    | llm
    llm is a Chat or LLM Runnable (e.g. a ChatOpenAI or ChatOllama instance). It takes the one-big prompt from custom_rag_prompt and returns the model’s completion (the final answer).
    
    """

    res = rag_chain.invoke(query)
    print(res)