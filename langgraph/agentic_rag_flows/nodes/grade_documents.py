from typing import Any, Dict

from agentic_rag_flows.chain import retrieval_grader
from agentic_rag_flows.chain.retrieval_grader import retriever_grader
from agentic_rag_flows.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """More actions
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search stateMore actions
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
