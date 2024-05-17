import asyncio
import json
import operator
import os
import asyncio
from typing import TypedDict, Annotated, Sequence, Dict, Literal

import yaml
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessageChunk
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langchain_core.tools import tool, BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

os.environ["LANGCHAIN_PROJECT"] = 'langgraph'

with open('api_key.yaml', 'r') as f:
    yaml_text = f.read()
api_key = yaml.load(yaml_text, Loader=yaml.FullLoader)['zhipu']


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


@tool
def multiply(a: int, b: int) -> int:
    """
    该函数用于计算两个整数的乘积。

    :param a: 整数参数a。
    :param b: 整数参数b。
    :return: 返回a和b的乘积，结果为整数。
    """
    return a * b


tools: list[BaseTool] = [multiply]
tool_executor = ToolExecutor(tools)

llm = ChatOpenAI(
    model="glm-4",
    openai_api_base='https://open.bigmodel.cn/api/paas/v4/',
    openai_api_key=api_key,
    streaming=True
)

llm = llm.bind_tools([convert_to_openai_function(t) for t in tools])


async def call_llm(state: AgentState) -> Dict[str, list]:
    messages = state.get('messages')
    response = await llm.ainvoke(messages)

    return {'messages': [response]}


def should_continue(state: AgentState) -> Literal['tools', '__end__']:
    message = state.get('messages')[-1]

    if 'tool_calls' not in message.additional_kwargs:
        return '__end__'

    return 'tools'


async def call_tool(state: AgentState):
    message = state.get('messages')[-1]
    func_call: dict = message.additional_kwargs.get('tool_calls')[0]

    action = ToolInvocation(
        tool=func_call.get('function').get('name'),
        tool_input=json.loads(func_call.get('function').get('arguments'))
    )

    response = await tool_executor.ainvoke(action)

    tool_message = ToolMessage(
        tool_call_id=func_call.get('id'),
        content=response
    )

    return {'messages': [tool_message]}


graph = StateGraph(AgentState)

graph.add_node('agent', call_llm)
graph.add_node('tools', call_tool)

graph.add_conditional_edges(
    'agent',
    should_continue,
)
graph.add_edge('tools', 'agent')

graph.set_entry_point('agent')
app = graph.compile()


# app.get_graph().draw_mermaid_png(
#     curve_style=CurveStyle.BASIS,
#     output_file_path='image/02StateGraph.png',
#     draw_method=MermaidDrawMethod.PYPPETEER
# )


async def test(question: str):
    inputs = {'messages': [HumanMessage(content=question)]}
    async for output in app.astream_log(inputs, stream_mode='updates'):
        for op in output.ops:
            if op["path"].startswith("/logs/") and op["path"].endswith(
                    "/streamed_output/-"
            ):
                if isinstance(op['value'], AIMessageChunk):
                    print(op['value'].content, end='')
                elif isinstance(op['value'], Dict) and 'messages' in op['value']:
                    print(op['value']['messages'][0].content, sep='\n', end='\n')


def main() -> None:
    asyncio.run(test('3乘4等于几?'))
    asyncio.run(test('蓝光对微藻有什么影响?'))


if __name__ == '__main__':
    main()
