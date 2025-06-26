from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import pandas as pd
from langchain_openai import ChatOpenAI
load_dotenv()

#llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=os.environ["OPENAI_KEY"], model="gpt-4o", temperature =0)
df = pd.read_csv("./data/salaries_2023.csv").fillna(value=0)
# print(df.head())

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool, PythonAstREPLTool

agent = create_pandas_dataframe_agent(
    llm=model,
    extra_tools=[],
    df=df,
    verbose=True,
    allow_dangerous_code=True
)
#res = agent.invoke("What is the average salary?")
#print(res)

#avg_salary = df['Base_Salary'].mean()
#print(f"The average salary is {avg_salary:.2f}")

CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""

QUESTION = "Which grade has the highest average base salary, and compare the average female pay vs male pay?"

#res = agent.invoke(QUESTION)
#print(res)

import streamlit as st

st.title("Database AI Agent with LangChain")

st.write("### Dataset Preview")
st.write(df.head())

# User input for the question
st.write("### Ask a Question")
question = st.text_input(
    "Enter your question about the dataset:",
    "Which grade has the highest average base salary, and compare the average female pay vs male pay?",
)

# Run the agent and display the result
if st.button("Run Query"):
    QUERY = question
    res = agent.invoke(QUERY)
    st.write("### Final Answer")
    st.markdown(res['output'])
