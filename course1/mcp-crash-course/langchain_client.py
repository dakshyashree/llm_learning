from langchain_mcp_adapters.client import MultiServerClient
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="llama3.1:8b")

async def main():
    print("Starting LangChain client...")
    #An asynchronous function (often called a coroutine in Python) is a function that can pause its execution at certain points to let other work run in the meantime, instead of blocking the entire program until it finishes.

if __name__ == "__main__":
     asyncio.run(main())