from typing import Annotated

from typing_extensions import TypedDict, Literal
from langchain_core.tools import InjectedToolCallId,tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv 
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt, Send
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage, SystemMessage
import operator
'''
创建工具调用工作流
'''
# 加载环境变量
load_dotenv(override=True)

# 模型信息
model = os.getenv("MODEL_NAME")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
llm = init_chat_model(model=model, base_url=base_url,api_key=api_key)

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

# 创建节点函数
def llm_call_generator(state: State):
    '''
    生成笑话
    '''
    if state.get("feedback"):
        msg = llm.invoke(f"写一个主题为 {state['topic']}的笑话，参考改进建议{state['feedback']}")
    else: 
        msg = llm.invoke(f"写一个主题为 {state['topic']}的笑话")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    '''
    评估笑话是否搞笑
    '''
    grade = evaluator.invoke(f"请评估这个笑话是否搞笑：{state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

# 创建条件边方法
def route_joke(state:State):
    '''
    根据评估结果，决定是否需要重新生成笑话
    '''
    if state["funny_or_not"] == "funny":
        return "Accepted"
    else:
        return "Rejected + Feedback"

# 创建图
graph_builder = StateGraph(State)

# 创建节点 
graph_builder.add_node("llm_call_generator", llm_call_generator)                
graph_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# 创建边
graph_builder.add_edge(START, "llm_call_generator")
graph_builder.add_edge("llm_call_generator", "llm_call_evaluator")
graph_builder.add_conditional_edges(
    "llm_call_evaluator", 
    route_joke,
    {"Accepted": END, "Rejected + Feedback": "llm_call_generator"}
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
