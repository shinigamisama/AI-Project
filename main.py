from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import AgentType, load_tools, initialize_agent, create_react_agent, AgentExecutor, ZeroShotAgent
from langchain_community.tools import Tool
from langchain.prompts import PromptTemplate
from datetime import datetime
import msal
import requests

app = FastAPI()


# Define a new tool that returns the current datetime
datetime_tool = Tool(
    name="datetime",
    func=lambda x: datetime.now().strftime("%A, %B %d, %Y"),
    description="Returns the current date, you have to use this tool everytime there is a date or a day in the question in order to calculate the correct date and use it to find the correct information",
)

llm = Ollama(
    model="llama3:instruct",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0.4,
    base_url='http://192.168.1.56:11434',
)

tools = load_tools(["ddg-search"],llm=llm)
tools.append(datetime_tool)
#tools.append(get_events_tool)

#prompt = hub.pull("hwchase17/react")

prompt_template = """Answer to the user as best and coplete as you can.

If the user ask for information about you or say hi, reply to the question straight forward without using any tools.
If you are not able to find a good answer with ddg-search tool then reply with the information you already have.
In case the user ask for code, always include the full code example in the final answer.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do (Can be used only 5 times)
Action: the action to take, must be one of [{tool_names}] (Only if needed)
Action Input: the input to the action (Can be used only 5 times)
Observation: the result of the action
Final Answer: the final answer to the original input question (Include always the full code found in observation part in the final answer)

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True,max_iterations=2)


class Query(BaseModel):
    question: str

@app.post("/query")
async def read_query(query: Query):
    response = agent_executor.invoke({"input":query.question})
    return {"response": response}
