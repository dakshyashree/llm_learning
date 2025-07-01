from dotenv import load_dotenv

load_dotenv()

from agentic_rag_flows.chain.retrieval_grader import GradeDocuments, retriever_grader
from agentic_rag_flows.ingestion import retriever
from pprint import pprint
from agentic_rag_flows.chain.generation import generation_chain
from agentic_rag_flows.chain.hallucination_grader import (
    hallucination_grader,
    GradeHallucinations,
)


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


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score
