import json
import os

import yaml

from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, MessageGraph

os.environ["LANGCHAIN_PROJECT"] = 'langgraph'

with open('api_key.yaml', 'r') as f:
    yaml_text = f.read()
ak = yaml.load(yaml_text, Loader=yaml.FullLoader)['qianfan']['ak']
sk = yaml.load(yaml_text, Loader=yaml.FullLoader)['qianfan']['sk']


@tool
def multiply(a: int, b: int) -> int:
    """
    该函数用于计算两个整数的乘积。

    :param a: 整数参数a。
    :param b: 整数参数b。
    :return: 返回a和b的乘积，结果为整数。
    """
    return a * b


model = QianfanChatEndpoint(
    model='ERNIE-Bot-4',
    qianfan_ak=ak,
    qianfan_sk=sk
)

tool_model = model.bind_tools(tools=[convert_to_openai_tool(multiply)])


def invoke_model(state: list[BaseMessage]) -> BaseMessage:
    return tool_model.invoke(state)


def invoke_tool(state: list[BaseMessage]) -> BaseMessage:
    tool_calls = state[-1].additional_kwargs.get('tool_calls', [])
    multiply_call: dict = None

    for tool_call in tool_calls:
        if tool_call.get('function').get('name') == 'multiply':
            multiply_call = tool_call

    if multiply_call is None:
        raise Exception('error')

    result = multiply.invoke(
        json.loads(multiply_call.get('function').get('arguments'))
    )

    return ToolMessage(
        # tool_call_id=multiply_call.get('function').get('id'),
        tool_call_id=state[-1].id,
        content=result
    )


def router(state: list[BaseMessage]) -> str:
    tool_calls: dict = state[-1].additional_kwargs.get('function_call', {})

    if 'name' in tool_calls:
        return 'multiply'
    else:
        return 'end'


graph = MessageGraph()

graph.add_node('start', invoke_model)
graph.add_node('multiply', invoke_tool)

graph.add_conditional_edges('start', router, {
    'multiply': 'multiply',
    'end': END
})
graph.add_edge('multiply', END)

graph.set_entry_point('start')
runnable = graph.compile()

res1 = runnable.invoke(HumanMessage('3乘4等于几？'))
res2 = runnable.invoke(HumanMessage('蓝光对微藻有什么影响？'))
