import os

import yaml

from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

os.environ["LANGCHAIN_PROJECT"] = 'langgraph'

with open('api_key.yaml', 'r') as f:
    yaml_text = f.read()
ak = yaml.load(yaml_text, Loader=yaml.FullLoader)['qianfan']['ak']
sk = yaml.load(yaml_text, Loader=yaml.FullLoader)['qianfan']['sk']

model = QianfanChatEndpoint(
    modle='ERNIE-Bot-4',
    qianfan_ak=ak,
    qianfan_sk=sk
)

graph = MessageGraph()

graph.add_node('start', model)
graph.add_edge('start', END)

graph.set_entry_point('start')
runnable = graph.compile()

res = runnable.invoke(HumanMessage('1加1等于几？'))
