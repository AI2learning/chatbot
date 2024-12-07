import faiss
import numpy as np
from PyPDF2 import PdfReader
from zhipuai import ZhipuAI
from langchain.text_splitter import CharacterTextSplitter
import json
import os
import glob

import config.static_cfg as cfg

vector_db_path = os.path.join(cfg.base_path, 'server', 'vector_db.index')
metadata_path = os.path.join(cfg.base_path, 'server', 'metadata.json')
pdf_docs_path = os.path.join(cfg.base_path, 'server', 'developers-pdf/*.pdf')

api_key = cfg.api_key
client = ZhipuAI(api_key=api_key)

# 加载/保存向量数据库的函数
def load_vector_db(index_path=vector_db_path, metadata_path=metadata_path):
    print("Loading vector database and metadata...")
    index = faiss.read_index(index_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata

def save_vector_db(index, metadata, index_path=vector_db_path, metadata_path=metadata_path):
    print("Saving vector database and metadata...")
    faiss.write_index(index, index_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def zhipu_embedding(text_chunks):
    response = client.embeddings.create(
        model="embedding-3",  # 填写需要调用的模型编码
        input=text_chunks,
    )
    embeddings = [data.embedding for data in response.data]
    return embeddings

def batch_process_embeddings(text_chunks, batch_size=10):
    all_embeddings = []
    all_metadata = []

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        batch_embeddings = zhipu_embedding(batch)
        all_embeddings.extend(batch_embeddings)
        all_metadata.extend([{"data": chunk} for chunk in batch])

        print(f"Processed batch {i // batch_size + 1}, total chunks processed: {i + len(batch)}")

    return all_embeddings, all_metadata