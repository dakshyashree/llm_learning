from langchain_experimental.agents import create_csv_agent
from langchain_ollama import ChatOllama
import pandas as pd

# 1️⃣ Instantiate your LLM
llm = ChatOllama(model="llama3")

# 2️⃣ Create the CSV agent, opting in to dangerous code execution
agent = create_csv_agent(
    llm=llm,  # pass the instance, not the class
    path=r"C:\Users\daksh\Downloads\sample_employee_records.csv",
    verbose=True,
    allow_dangerous_code=True,
)

# 3️⃣ If you want to peek at the CSV yourself:
df = pd.read_csv(r"C:\Users\daksh\Downloads\sample_employee_records.csv")
print(df.head())

# 4️⃣ And to ask the agent a question:
response = agent.invoke(
    "give me the list of names whose last name is Anderson with count?"
)
print("Agent answer:", response)
