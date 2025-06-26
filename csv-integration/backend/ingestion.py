from langchain_ollama import ChatOllama
from langchain_experimental.agents import create_pandas_dataframe_agent

from langchain_experimental.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI  # or your preferred LLM
import pandas as pd

# 1. Instantiate your LLM and agent
llm = ChatOllama(model="llama3", temperature=0)
agent = create_csv_agent(
    llm=llm,
    path=r"C:\Users\daksh\Downloads\sample_employee_records.csv",
    verbose=True,
    allow_dangerous_code=True,
)

# 2. Tell the agent exactly which tool to run and with what code
prompt = """
Action: python_repl_ast
Input: df[df["Name"].str.endswith(" Anderson")]["Name"].tolist()
"""

result = agent.invoke(prompt, handle_parsing_errors=True)
print("Names:", result)

# 3. If you also need the count:
prompt_count = """
Action: python_repl_ast
Input: len(df[df["Name"].str.endswith(" Anderson")])
"""

count = agent.invoke(prompt_count, handle_parsing_errors=True)
print("Count:", count)
