import os
import time
from datetime import datetime
import random
import numpy as np
import http.client
import json
import uuid
from openai import OpenAI
import json
import requests
from dataclasses import asdict, dataclass
import streamlit as st
from zhipuai import ZhipuAI

import config.static_cfg as cfg
from utils import load_vector_db, save_vector_db, get_pdf_text, get_text_chunks
from utils import zhipu_embedding, batch_process_embeddings
from loguru import logger

log_file_path = os.path.join(cfg.base_path, "logs", "app.log")
logger.add(log_file_path)

client = ZhipuAI(api_key=cfg.api_key)

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    use_search: bool = True
    repetition_penalty: float = 1.005


def on_btn_click():
    del st.session_state.messages


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=32000,
                               value=32000)
        top_p = st.slider('Top P', 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.7, step=0.01)
        use_search = st.toggle("是否使用搜索功能", value=False)
        st.button('Clear Chat History', on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature,
                                         use_search=use_search)

    return generation_config



model = OpenAI(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
    )


def generate_deepseek(prompt, history):
    history.append({'role': 'user', 'content': prompt})
    response = model.chat.completions.create(
        model="deepseek-coder",  # 填写需要调用的模型名称
        messages=history,
        stream=True,
    )
    return response

GENERATE_TEMPLATE = """
你是一位企业的智能客服，名字叫小海，如果用户输入的问题命中知识库，参照知识库中的内容进行回答，需要给出参考链接。
如果用户的问题不是，只是闲聊，请使用友善的语句与用户进行交流，不需要给出参考链接。
如果是你不能解决的问题，请回答友善的回答我不知道，不需要给出参考链接。

问题: "{}"

参考资料: "{}"
"""

index, metadata = load_vector_db()

def get_reference(query):
    # try:
    # 查询示例
    # query = "G123平台是一个什么平台？"
    query_vector = zhipu_embedding([query])
    query_vector = np.array(query_vector)

    num_results = 3  # 返回前 3 个结果
    distances, indices = index.search(query_vector, num_results)

    logger.debug(f"Distances: {distances}")
    logger.debug(f"Indices: {indices}")

    references = []
    for idx in indices[0]:
        references.append(metadata[idx])
    return references
        # print(f"Metadata for index {idx}: {metadata[idx]}")
    # except:
    #     return []

def main():

    st.title('智能客服')

    generation_config = prepare_generation_config()
    print (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), generation_config)

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Accept user input
    if query := st.chat_input('Hello!'):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(query)

        references = get_reference(query)
        prompt = GENERATE_TEMPLATE.format(query,references)
        logger.debug(f"final prompt: {prompt}")

        # 是否启用搜索功能
        if generation_config.use_search:
            pass


        with st.chat_message('robot'):
            message_placeholder = st.empty()
            ai_message = []
            history = st.session_state.messages.copy()
            for cur_response in generate_deepseek(prompt, history):
                # Display robot response in chat message container
                ai_message.append(cur_response.choices[0].delta.content)            
                message_placeholder.markdown(''.join(ai_message) + '▌') 

            message_placeholder.markdown(''.join(ai_message))


        # Add user message to chat histosry
        st.session_state.messages.append({
            'role': 'user',
            'content': query,
        })

        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'assistant',
            'content': ''.join(ai_message),  
        })
        print ("st.session_state.messages: ", st.session_state.messages)


if __name__ == '__main__':

    main()
