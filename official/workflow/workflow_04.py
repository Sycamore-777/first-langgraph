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
创建动态工作流
'''
# 加载环境变量
load_dotenv(override=True)

# 模型信息
model = os.getenv("MODEL_NAME")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
llm = init_chat_model(model=model, base_url=base_url,api_key=api_key)

# 创建规划器（planner）的输出结构
class Section(BaseModel):
    name: str = Field(description="报告中本章名字")
    description: str = Field(description="报告中本章描述.")

class Sessions(BaseModel):
    sections: list[Section] = Field(description="列举报告章节.")

# 升级模型
planner = llm.with_structured_output(Sessions) # 增加结构化输出

# 创建图模板
class State(TypedDict):
    topic: str # 报告主题
    sections: list[Section] # 报告章节列表
    completed_sections: Annotated[list, operator.add] # 所有的workers都并行地写在这个key中
    final_report: str # 最终报告

# 创建工作器
class WorkerState(TypedDict):
    section: Section # 报告章节
    completed_sections: Annotated[list, operator.add]

# 创建节点函数
def orchestrator(state: State): 
    '''
    为报告生成计划的编排器
    '''
    report_sections= planner.invoke(
        [
            SystemMessage(content="为报告生成一个计划。"),
            HumanMessage(content=f"报告的主题是：{state['topic']}")
        ]
    )
    return {
        "sections": report_sections.sections
        }

def llm_call(state: WorkerState):
    '''
    根据报告章节生成内容,就是workers
    '''
    section = llm.invoke(
        [
            SystemMessage(content="根据提供的章节名称和描述编写章节内容。每个部分不包含前言。使用Markdown格式。"),
            HumanMessage(content=f"章节名称：{state['section'].name}，章节描述：{state['section'].description}"),
        ]
    )
    return {
        "completed_sections": [section.content]
    }

def synthesize(state: State):
    '''
    将所有章节合并为最终报告
    '''
    # # 列举当前状态下已完成的章节
    # completed_sections = state["completed_sections"]

    # # 将完成的章节格式化为字符串，用作最终章节的上下文
    # completed_report_sections  = "\n\n---\n\n".join(completed_sections)
    # return {"final_report": completed_report_sections}
        # 拿到章节顺序
    sections = state["sections"]
    completed_sections = state["completed_sections"]

    # 将 completed_sections 变为 dict: 标题 -> 内容
    section_map = {}
    for content in completed_sections:
        for section in sections:
            if section.name in content:
                section_map[section.name] = content
                break

    # 按照 sections 顺序构造最终文本
    ordered_contents = [
        section_map.get(section.name, f"【缺失章节：{section.name}】") for section in sections
    ]

    completed_report_sections = "\n\n---\n\n".join(ordered_contents)
    return {"final_report": completed_report_sections}

# 创建分配方法
def assign_workers(state:State):
    '''
    为每个章节分配一个worker
    '''
    # 为每个章节分配一个worker,通过Send api启动与运行
    return [Send("llm_call", {"section": s}) for s in state["sections"]]
    # return [Send("llm_call", {"section": s}) for s in state["sections"]]


# 创建图
graph_builder = StateGraph(State)

# 创建节点 (编排任务→分配任务→workers→最终合成)
graph_builder.add_node("llm_call", llm_call)                # workers
graph_builder.add_node("synthesize", synthesize)            # 合成
graph_builder.add_node("orchestrator", orchestrator)        # 编排任务

# 创建边
graph_builder.add_edge(START, "orchestrator")
graph_builder.add_conditional_edges(
    "orchestrator", 
    assign_workers,
    {"llm_call": "llm_call"}
)
graph_builder.add_edge("llm_call", "synthesize")
graph_builder.add_edge("synthesize", END)

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
