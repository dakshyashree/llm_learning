from typing import Any, Dict
from langchain.schema import (
    Document,
)  # used to convert the tavily search results into langchain document
from langchain_tavily import TavilySearch
from agentic_rag_flows.state import GraphState

web_search_tool = TavilySearch(max_results=3)
from dotenv import load_dotenv

load_dotenv()


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---Web Search---")
    question = state["question"]
    documents = state["documents"]

    tavily_searches = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_searches]
    )

    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
