import logging
import random
import os

# api version
API_VER = 'v1'

# 设置环境变量
base_path = os.path.dirname(os.path.dirname(__file__))
print ("cfg base_path: ", base_path)

# 智谱API
api_key = ""

# openai接口的key和base_url
llm_api_key=""
llm_base_url='https://api.deepseek.com'