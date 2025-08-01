from typing import Annotated

from typing_extensions import TypedDict
from langchain_core.tools import InjectedToolCallId,tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv 
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
'''
创建具有条件分支的工作流
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
    topic: str
    joke: str
    improved_joke: str
    final_joke: str
    messages: Annotated[list, add_messages]

# 节点定义
def generate_joke(state: State):
    '''
    生成笑话
    '''
    msg = llm.invoke(f"生成一个{state['topic']}主题的笑话")
    return {"joke": msg.content}

def check_punchline(state: State):
    """Gate function to check if the joke has a punchline"""

    # Simple check - does the joke contain "?" or "!"
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"

def improve_joke(state: State):
    """Second LLM call to improve the joke"""

    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}

def polish_joke(state: State):
    """Third LLM call for final polish"""

    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}

# 构建工作流
graph_builder = StateGraph(State)

# 构建节点
graph_builder.add_node("generate_joke", generate_joke)
graph_builder.add_node("improve_joke", improve_joke)
graph_builder.add_node("polish_joke", polish_joke)

# 构建边
graph_builder.add_edge(START, "generate_joke")
graph_builder.add_conditional_edges("generate_joke", check_punchline,{"Pass": END, "Fail": "improve_joke"})
graph_builder.add_edge("improve_joke", "polish_joke")
graph_builder.add_edge("polish_joke", END)

# 编译工作流
graph = graph_builder.compile()

if __name__ == "__main__":
    # 可视化
    from IPython.display import Image, display
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass
    try:
        img_bytes = graph.get_graph().draw_mermaid_png()
        with open("chatbot_graph.png", "wb") as f:
            f.write(img_bytes)
        print("流程图已保存为 chatbot_graph.png")
    except Exception:
        print("可视化失败，可能缺少依赖。")

    # 推理
    state = graph.invoke({"topic": "喜剧"})
    print("Initial joke:")
    print(state["joke"])
    print("\n--- --- ---\n")
    if "improved_joke" in state:
        print("Improved joke:")
        print(state["improved_joke"])
        print("\n--- --- ---\n")

        print("Final joke:")
        print(state["final_joke"])
    else:
        print("Joke failed quality gate - no punchline detected!")