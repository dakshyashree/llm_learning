from typing import Any, Dict
from agentic_rag_flows.chain.generation import generation_chain
from agentic_rag_flows.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---Generate---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
