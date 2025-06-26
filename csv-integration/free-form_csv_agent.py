from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import pandas as pd
from langchain_openai import ChatOpenAI
load_dotenv()
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool, PythonAstREPLTool


#llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=os.environ["OPENAI_KEY"], model="gpt-4o", temperature =0)
df = pd.read_csv("./data/salaries_2023.csv").fillna(value=0)
# print(df.head())

agent = create_pandas_dataframe_agent(
    llm=model,
    extra_tools=[],
    df=df,
    verbose=True,
    allow_dangerous_code=True
)

import streamlit as st

st.title("Database AI Agent with LangChain")

st.write("### Dataset Preview")
st.write(df.head())

# User input for the question
st.write("### Ask a Question")
question = st.text_input(
    "Example questions->"
    "Enter your question about the dataset:",
    "Which grade has the highest average base salary, and compare the average female pay vs male pay?",
)

# Run the agent and display the result
if st.button("Run Query"):
    try:
        answer = agent.invoke(question, handle_parsing_errors=True)
    except Exception:
        # fallback to plain LLM
        prompt = PromptTemplate.from_template(
            "Here is a snapshot of the data:\n{df_head}\n\nQuestion: {question}\nAnswer:"
        )
        llm = ChatOpenAI(api_key=os.environ["OPENAI_KEY"], model="gpt-4o", temperature =0)
        chain = prompt | llm
        inputs = {
            "df_head": df.head().to_markdown(),
            "question": question
        }
        answer = chain.invoke(inputs)
    st.markdown(answer['output'])
