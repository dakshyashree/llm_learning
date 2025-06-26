from typing import Union, List

from dotenv import load_dotenv
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, tool
from langchain_ollama import ChatOllama
from langchain.tools import Tool, tool

load_dotenv()


@tool  # registers as langchain tool
# LangChain will include its docstring under “Description:” when you call render_text_description(tools).
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text-length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping the non-alphabetic characters just in case
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("hello Langchain")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial( #pre-fills the {tools} and {tool_names} slots so the LLM sees the tool list in its system prompt.
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm = ChatOllama(model="deepseek-R1", stop=["\nObservation"])
    # We set stop=["\nObservation"] so that as soon as the agent writes out “Observation: …”, the LLM will stop and return to LangChain, which then executes the tool.
    intermediate_steps = []
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),#format_log_to_str-> passes only strings to the llm
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of 'DOG' in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
        intermediate_steps.append(
            (agent_step, str(observation))
        )  # has logs on both its reasoning engine history and the result of tools executiong
# wont work with deepseek only with openai
