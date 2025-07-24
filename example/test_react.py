import requests,json
import os
from dotenv import load_dotenv 
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import InMemorySaver
memory = MemorySaver()

# 加载环境变量
load_dotenv(override=True)

# 自定义工具集合
class Write_Query(BaseModel):
    content: str = Field(description="需要写入文档的具体内容")
class Get_Weather(BaseModel):
    city: str = Field(description="需要查询天气的城市名称")

@tool(args_schema=Write_Query)
def write_query(content: str) -> str:
    """
    将指定的内容写入文档
    :param content: 需要写入文档的具体内容
    :return: 写入成功标志
    """
    return "写入成功"

@tool(args_schema=Get_Weather)
def get_weather(city: str) -> str:
    """
    获取指定城市的天气情况
    :param city: 需要查询天气的城市名称
    :return: 天气情况
    """
    return f"{city}天气晴朗"

# 初始化Tavily搜索工具
tavily_search_tool = TavilySearch(max_results=5,topic="general",tavily_api_key=os.getenv("TAVILY_KEY"))

# 大模型架加载
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
model = init_chat_model(model="deepseek-chat", base_url=BASE_URL,api_key=API_KEY)

# 工具集合
tools = [write_query,get_weather,tavily_search_tool]

# 创建一个内存保存器
checkpoint = InMemorySaver()
config = {
    "configurable": {
        "thread_id": "1",  # 线程ID
    },
    "recursion_limit": 10,
    "stream": True
}

# 智能体构建
agent = create_react_agent(model=model, tools=tools, checkpointer=checkpoint)

# 使用智能体进行对话
while True:
    user_input = input("🙋：")
    if user_input.lower() == 'quit':
        break
    response = agent.invoke(
        {
            "messages": [
                {"role": "system", "content": "你是一个乐于助人的助手，请根据用户的问题给出回答"},
                {"role": "user", "content": user_input}
            ]
        },
        # {"recursion_limit":10}, # 设置递归限制,包括用户消息，工具调用，工具响应，最终响应等
        config=config  # 配置参数
    )
    print(f'🤖：{response["messages"][-1].content}')


# 打印响应结果
print(response)
print("=="*30)
print(response["messages"][-1].content)