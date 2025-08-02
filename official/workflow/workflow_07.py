from typing import Annotated

from typing_extensions import TypedDict, Literal
from langchain_core.tools import InjectedToolCallId,tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv 
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt, Send
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import operator
'''
创建评估优化工作流
'''
# 加载环境变量
load_dotenv(override=True)

# 模型信息
model = os.getenv("MODEL_NAME")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
llm = init_chat_model(model=model, base_url=base_url,api_key=api_key)

# 创建工具
# 定义MCP
client = MultiServerMCPClient(
    {

        "detect": {
            # Ensure you start your weather server on port 8000
            "url": "http://localhost:9000/mcp",
            "transport": "streamable_http",
        }
    }
)

# client = MultiServerMCPClient(
#     {

#         "detect": {
#             # Ensure you start your weather server on port 8000
#             "url": "http://localhost:8088/sse",
#             "transport": "sse",
#         }
#     }
# )

async def weather():
    tools = await client.get_tools()
    graph = create_react_agent(model=llm, tools=tools)

    weather_response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "今天云南天气如何?"}]}
    )
    print(weather_response["messages"][-1].content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(weather())