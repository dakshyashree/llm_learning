from typing import Any, Dict
from agentic_rag_flows.state import GraphState
from agentic_rag_flows.ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---Retrieve---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
