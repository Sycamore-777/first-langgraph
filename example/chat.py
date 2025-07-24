import os
from dotenv import load_dotenv 
from openai import OpenAI

load_dotenv()
# load_dotenv(override=True)
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
# 初始化DeepSeek的API客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 调用DeepSeek的API，生成回答
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "你是乐于助人的助手，请根据用户的问题给出回答"},
        {"role": "user", "content": "你好，请你介绍一下你自己。"},
    ],
)

# 打印模型最终的响应结果
print(response.choices[0].message.content)