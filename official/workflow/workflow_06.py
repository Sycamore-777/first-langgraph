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
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}

# 创建图模板
class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

# 创建评估器的输出模板
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(description="判断这个笑话是否搞笑，返回 funny 或者 not funny")
    feedback: str = Field(description="如果这个笑话不好笑，请给出改进建议")

# 改进大模型，增加结构化输出，用于作为评估器模型
evaluator = llm.with_structured_output(Feedback)
llm_with_tools = llm.bind_tools(tools)
# 创建节点函数
def llm_call(state: MessagesState):
    '''
    大模型自主决定是否需要调用工具.大模型节点
    '''
    return {
        "messages":[
            llm_with_tools.invoke(
                [
                    SystemMessage(content="你是一个乐于助人的助手，负责算数计算"),
                ]
                + state["messages"]
            )
        ]
    }

def tool_node(state: MessagesState):
    '''
    执行工具调用节点
    '''
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState):
    '''
    判断是否需要继续调用大模型
    '''
    message = state["messages"]
    last_message = message[-1]
    if last_message.tool_calls:
        return "Action"
    else:
        return END

# 创建图
graph_builder = StateGraph(MessagesState)

# 创建节点 
graph_builder.add_node("llm_call", llm_call)                
graph_builder.add_node("tool_node", tool_node)

# 创建边
graph_builder.add_edge(START, "llm_call")
graph_builder.add_conditional_edges(
    "llm_call", 
    should_continue,
    {"Action": "tool_node", END: END}
)

# 编译图
graph = graph_builder.compile()

# 主函数，显示图片，工作流调用
if __name__ == "__main__":
    # 可视化
    from IPython.display import Image, display
    try:
        display(Image(graph.get_graph().draw_png()))
    except ImportError:
        print(
            "You likely need to install dependencies for pygraphviz, see more here https://github.com/pygraphviz/pygraphviz/blob/main/INSTALL.txt"
        )
    # from IPython.display import Image, display
    # try:
    #     display(Image(graph.get_graph().draw_mermaid_png()))
    # except Exception:
    #     # This requires some extra dependencies and is optional
    #     pass
    # try:
    #     img_bytes = graph.get_graph().draw_mermaid_png()
    #     with open("chatbot_graph.png", "wb") as f:
    #         f.write(img_bytes)
    #     print("流程图已保存为 chatbot_graph.png")
    # except Exception:
    #     print("可视化失败，可能缺少依赖。")

    # 推理
    state = graph.invoke({"topic": "写一个关于LLM的Scaling laws 的报告"})
    print(state["output"])
