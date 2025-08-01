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
from langgraph.types import Command, interrupt
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage, SystemMessage
'''
创建具有路由结构的工作流
'''
# 加载环境变量
load_dotenv(override=True)

# 模型信息
model = os.getenv("MODEL_NAME")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
llm = init_chat_model(model=model, base_url=base_url,api_key=api_key)

# 创建输出模板
class Route(BaseModel):
    step: Literal["joke", "story", "poem"] = Field(None, description= "路由节点的下一步")

# 升级模型
router= llm.with_structured_output(Route) # 增加结构化输出

# 创建图模板
class State(TypedDict):
    input: str
    decision: str
    output: str

# 创建节点函数
def llm_call_1(state: State):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(state["input"])
    return {"output": msg.content}


def llm_call_2(state: State):
    """Second LLM call to generate story"""

    msg = llm.invoke(state["input"])
    return {"output": msg.content}


def llm_call_3(state: State):
    """Third LLM call to generate poem"""

    msg = llm.invoke(state["input"])
    return {"output": msg.content}


def llm_call_router(state: State):
    '''
    使用大模型辅助进行路由结果选择，并且输出结构化的路由结果
    '''
    """Combine the joke and story into a single output"""
    decision= router.invoke([
        {"role":"system", "content":"基于用户的输入，请从 joke、story、poem 中选择一个作为下一步的输出"},
        {"role":"user", "content":state["input"]}
    ])
    # TODO 测试一下这两种方式有什么区别
    # decision = router.invoke(
    #     [
    #         SystemMessage(
    #             content="Route the input to story, joke, or poem based on the user's request."
    #         ),
    #         HumanMessage(content=state["input"]),
    #     ]
    # )
    return {"decision": decision.step}

# 创建路由条件
def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"

# 创建图
graph_builder = StateGraph(State)

# 创建节点
graph_builder.add_node("llm_call_router",llm_call_router)
graph_builder.add_node("llm_call_1",llm_call_1)
graph_builder.add_node("llm_call_2",llm_call_2)
graph_builder.add_node("llm_call_3",llm_call_3)

# 创建边
graph_builder.add_edge(START, "llm_call_router")
graph_builder.add_conditional_edges(
    "llm_call_router", 
    route_decision,
    {"llm_call_1": "llm_call_1", 
     "llm_call_2": "llm_call_2", 
     "llm_call_3": "llm_call_3"})
graph_builder.add_edge("llm_call_1", END)
graph_builder.add_edge("llm_call_2", END)
graph_builder.add_edge("llm_call_3", END)

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
    state = graph.invoke({"input": "写一个关于朱棣的笑话"})
    print(state["output"])
