from dotenv import load_dotenv
from agentic_rag_flows.chain.retrieval_grader import GradeDocuments, retriever_grader
from agentic_rag_flows.ingestion import retriever
from pprint import pprint
from agentic_rag_flows.chain.generation import generation_chain

load_dotenv()


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retriever_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    res: GradeDocuments = retriever_grader.invoke(
        {"question": "how to make pizza", "document": doc_txt}
    )
    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"content": docs, "question": question})
    pprint(generation)
