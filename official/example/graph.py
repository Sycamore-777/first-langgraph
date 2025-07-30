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
from langchain_core.messages import ToolMessage

# 加载环境变量
load_dotenv(override=True)

############### 工具设置区 ###############
from langchain_tavily import TavilySearch
# 搜索引擎
tavily_search_tool = TavilySearch(max_results=2)
# print(tavily_search_tool.invoke("What's a 'node' in LangGraph?"))

# 打断
@tool
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)


# 工具汇总
# tools = [human_assistance]
tools = [tavily_search_tool,human_assistance]
############### 图构建区 ###############
# 创建图
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    name: str
    birthday: str
graph_builder = StateGraph(State)
print(graph_builder)

# 模型信息
model = os.getenv("MODEL_NAME")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
llm = init_chat_model(model=model, base_url=base_url,api_key=api_key)
llm_with_tools = llm.bind_tools(tools=tools) # 将工具添加到大模型中
# 构建检查点器 TODO 后续有可能使用SqliteSaver 或 PostgresSaver 并连接数据库
memory = MemorySaver()

# 构建节点
def chatbot(state: State):
    '''构建聊天节点 '''
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1

    return {"messages": [message]}
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)           # 创建工具调用节点
graph_builder.add_node("tools", tool_node)
# 构建边
graph_builder.add_edge(START, "chatbot")    # 创建开始节点，并连接到chatbot节点
graph_builder.add_edge("chatbot", END)      # 创建结束节点，并连接到chatbot节点
graph_builder.add_conditional_edges(        # 连接工具节点和chatbot节点
    "chatbot",
    tools_condition,  # Routes to "tools" or "__end__"
    {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "chatbot")

#编译图
graph = graph_builder.compile(checkpointer=memory)
# graph = graph_builder.compile()

if __name__ == "__main__":
    #可视化图
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


    ############### 聊天内容更新与解析 ###############
    config = {"configurable": {"thread_id": "1"}}
    def stream_graph_updates(user_input: str):
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config = config,
            stream_mode="values",
        )
        for event in events:
            event["messages"][-1].pretty_print()
    # def stream_graph_updates(user_input: str):
    #     # for event in graph.stream({"messages": [{"role": "user", "content": user_input}]},):
    #     for event in graph.stream({"messages": [{"role": "user", "content": user_input}]},
    #                               config=config,
    #                               stream_mode="values"):
    #         for value in event.values():
    #             print("Assistant:", value["messages"][-1].content)


    ############### 聊天机器人 ###############
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except Exception as e:
            # fallback if input() is not available
            # user_input = "What do you know about LangGraph?"
            # print("User: " + user_input)
            # stream_graph_updates(user_input)
            print(f"遇到错误：{e}")
            break