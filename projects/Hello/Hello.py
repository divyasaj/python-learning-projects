# This is my first program to create a AL Agent

from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

from langchain_community.utilities import SerpAPIWrapper
from langchain_community.agent_toolkits.load_tools import Tool, load_tools

from langchain.agents import AgentType, initialize_agent

gpt4 = ChatOpenAI(model="gpt-4", temperature=0)
#print(gpt4.invoke("Who is the NBA all-time leading scorer? What's his total points to the power of 0.42?"))
#print(gpt4.invoke("who is the leading wicket taker in test cricket with the best average?"))


search = SerpAPIWrapper()
# create the serp tool
serp_tool = Tool(
  name="Search",
  func=search.run,
  description="useful for when you need to answer questions about current events. You should ask targeted questions",
)

# Initialize tools with calculator and the model
gpt4_tools = load_tools(["llm-math"], llm=gpt4)
# add the serp tool
gpt4_tools = gpt4_tools + [serp_tool]
# initialize GPT-4 Agent
gpt4agent = initialize_agent(
  gpt4_tools,
  ChatOpenAI(model="gpt-4", temperature=0),
  agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True,
)
gpt4agent.invoke("who is the leading wicket taker in test cricket with the best average?")

print('Successful execution')
