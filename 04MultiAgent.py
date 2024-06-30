import functools
import operator
import os
import yaml
from typing import Annotated, TypedDict, Sequence, Literal

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_core.tools import tool, BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode

with open('api_key.yaml', 'r') as f:
    yaml_text = f.read()
llm_url = yaml.load(yaml_text, Loader=yaml.FullLoader)['zhipu']['url']
llm_api = yaml.load(yaml_text, Loader=yaml.FullLoader)['zhipu']['ak']
tavily_api = yaml.load(yaml_text, Loader=yaml.FullLoader)['tavily']

os.environ["LANGCHAIN_PROJECT"] = 'langgraph'


def create_agent(llm: ChatOpenAI, tools: list[BaseTool], system_message: str):
    """创建一个智能体."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一位能干的AI助手，并与其他助手共同工作。"
                "使用提供的工具来解决问题。"
                "如果你不能完善的回答，没关系，另一位拥有不同工具的助手会在你停下的地方继续工作。"
                "只需要执行你能做的来推动工作进展。"
                "如果你或其他助手得到了最终答案或可交付的结果，在回答前加上“FINAL ANSWER”，让团队知道该停止工作了。"
                "你可以使用以下工具：{tool_names}。\n{system_message}"

            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([_tool.name for _tool in tools]))
    return prompt | llm.bind_tools(tools)


tavily_tool = TavilySearchResults(
    max_results=5,
    api_wrapper=TavilySearchAPIWrapper(
        tavily_api_key=tavily_api
    ),
    description='一个为全面、准确和可信赖的结果而优化的搜索引擎。'
                '当你需要回答一些有关当下事件的问题时，它非常有用。'
                '输入内容是一个搜索查询语句。'
)

# Warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()


@tool
def python_repl(
        code: Annotated[str, "The python code to execute to generate your chart."],
):
    """使用这个工具来执行python代码。 如果你想看到某个变量的值,
    你需要使用 `print(...)`函数来打印它。这对于用户是可见的。"""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"执行成功 :\n```python\n{code}\n```\nStdout: {result}"
    return (
            result_str + "\n\n如果已经完成了全部任务, 返回“FINAL ANSWER”."
    )


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Helper function to create a node for a given agent
def agent_node(state: AgentState, agent: RunnableSerializable[dict, BaseMessage], name: str):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"


def build_graph() -> CompiledGraph:
    llm = ChatOpenAI(
        model="glm-4",
        openai_api_base=llm_url,
        openai_api_key=llm_api
    )

    research_agent = create_agent(
        llm,
        [tavily_tool],
        system_message="你需要为对话生成助手提供它所需要的准确的数据。",
    )
    research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

    # chart_generator
    chart_agent = create_agent(
        llm,
        [python_repl],
        system_message="用户可以看到你显示的任何图表。",
    )
    chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

    tools = [tavily_tool, python_repl]
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("Researcher", research_node)
    workflow.add_node("chart_generator", chart_node)
    workflow.add_node("call_tool", tool_node)

    workflow.add_conditional_edges(
        "Researcher",
        router,
        {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
    )
    workflow.add_conditional_edges(
        "chart_generator",
        router,
        {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
    )

    workflow.add_conditional_edges(
        "call_tool",
        # Each agent node updates the 'sender' field
        # the tool calling node does not, meaning
        # this edge will route back to the original agent
        # who invoked the tool
        lambda x: x["sender"],
        {
            "Researcher": "Researcher",
            "chart_generator": "chart_generator",
        },
    )
    workflow.set_entry_point("Researcher")
    graph = workflow.compile()
    return graph


def main() -> None:
    graph = build_graph()
    graph.get_graph().draw_mermaid_png(
        curve_style=CurveStyle.BASIS,
        output_file_path='image/04MultiAgent.png',
        draw_method=MermaidDrawMethod.PYPPETEER
    )

    graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Fetch the UK's GDP over the past 5 years,"
                            " then draw a line graph of it."
                            " Once you code it up, finish."
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    )


if __name__ == '__main__':
    main()
