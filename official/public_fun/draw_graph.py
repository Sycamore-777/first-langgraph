import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from io import BytesIO
from typing import Annotated

from typing_extensions import TypedDict
from langchain_core.tools import InjectedToolCallId, tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

"""
创建并行的工作流
"""
# 加载环境变量
load_dotenv(override=True)

# 模型信息
model = os.getenv("MODEL_NAME")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
llm = init_chat_model(model=model, base_url=base_url, api_key=api_key)


# 创建图模板
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


# 创建节点函数
def call_llm_1(state: State):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"写一个主题为 {state['topic']}的笑话")
    return {"joke": msg.content}


def call_llm_2(state: State):
    """Second LLM call to generate story"""

    msg = llm.invoke(f"写一个主题为 {state['topic']}的故事")
    return {"story": msg.content}


def call_llm_3(state: State):
    """Third LLM call to generate poem"""

    msg = llm.invoke(f"写一个主题为 {state['topic']}的诗词")
    return {"poem": msg.content}


def aggregator(state: State):
    """Combine the joke and story into a single output"""

    combined = f"下面是关于 {state['topic']}主题的故事、笑话和诗歌!\n\n"
    combined += f"故事:\n{state['story']}\n\n"
    combined += f"笑话:\n{state['joke']}\n\n"
    combined += f"诗歌:\n{state['poem']}"
    return {"combined_output": combined}


# 创建图
graph_builder = StateGraph(State)

# 创建节点
graph_builder.add_node("call_llm_1", call_llm_1)
graph_builder.add_node("call_llm_2", call_llm_2)
graph_builder.add_node("call_llm_3", call_llm_3)
graph_builder.add_node("aggregator", aggregator)

# 创建边
graph_builder.add_edge(START, "call_llm_1")
graph_builder.add_edge(START, "call_llm_2")
graph_builder.add_edge(START, "call_llm_3")
graph_builder.add_edge("call_llm_1", "aggregator")
graph_builder.add_edge("call_llm_2", "aggregator")
graph_builder.add_edge("call_llm_3", "aggregator")
graph_builder.add_edge("aggregator", END)

# 编译图
graph = graph_builder.compile()

# 可视化图
graph_png = graph.get_graph().draw_mermaid_png()
img = mpimg.imread(BytesIO(graph_png), format="PNG")
plt.imshow(img)
plt.axis("off")
plt.show()
png_path = "./graph_image.png"
plt.imsave(png_path, img)
