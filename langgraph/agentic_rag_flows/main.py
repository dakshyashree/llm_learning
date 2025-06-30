from dotenv import load_dotenv

load_dotenv()
from agentic_rag_flows.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "what is agent memory?"}))
