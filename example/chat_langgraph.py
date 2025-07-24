from langchain.chat_models import init_chat_model
# from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# load_dotenv(override=True)
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
model = init_chat_model(model="deepseek-chat", base_url=BASE_URL,api_key=API_KEY)
# model = init_chat_model(model="deepseek-chat", model_provider="deepseek",api_key=DEEPSEEK_API_KEY)

question = "你好，请你介绍一下你自己。"

result = model.invoke(question)
print(result.content)