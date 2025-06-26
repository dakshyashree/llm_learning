from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM,ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num:float)->float:
    """
    param num:a number to triple
    return:the triple of the number
    """
    return float(num)*3

tools=[triple, TavilySearch(max_results=1)]

llm = OllamaLLM(model="deepseek-r1").bind_tools(tools)#function calling









