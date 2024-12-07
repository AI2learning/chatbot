import faiss
import numpy as np
from PyPDF2 import PdfReader
from zhipuai import ZhipuAI
from langchain.text_splitter import CharacterTextSplitter
import json
import os
import glob

import config.static_cfg as cfg
from utils import load_vector_db, save_vector_db, get_pdf_text, get_text_chunks
from utils import zhipu_embedding, batch_process_embeddings

vector_db_path = os.path.join(cfg.base_path, 'server', 'vector_db.index')
metadata_path = os.path.join(cfg.base_path, 'server', 'metadata.json')
pdf_docs_path = os.path.join(cfg.base_path, 'server', 'developers-pdf/*.pdf')

api_key = cfg.api_key
client = ZhipuAI(api_key=api_key)

def load_vectors():
    # 尝试加载现有的向量数据库和元数据，如果不存在则创建一个新的
    try:
        index, metadata = load_vector_db()
    except Exception as e:
        print(f"Could not load vector db or metadata: {e}. Creating new ones.")
        dimension = 2048  # 根据你的嵌入向量维度调整此值
        index = faiss.IndexFlatL2(dimension)
        metadata = []

        # 获取指定路径下的所有PDF文件
        
        pdf_docs = glob.glob(pdf_docs_path)
        pdf_docs = pdf_docs[:2]

        if not pdf_docs:
            print("没有找到任何PDF文件")
        else:
            print(f"找到 {len(pdf_docs)} 个PDF文件")

            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)

            # 批量处理文本块并获取嵌入向量和元数据
            embeddings, new_metadata = batch_process_embeddings(text_chunks, batch_size=10)

            if embeddings:
                print(f"Embedding size: {len(embeddings[0])}")

                # 更新索引和元数据
                embeddings = np.array(embeddings)
                index.add(embeddings)
                metadata.extend(new_metadata)

                # 在脚本结束时保存向量数据库和元数据
                save_vector_db(index, metadata)
                
    return index, metadata



def main():

    index, metadata = load_vectors()

    # 查询示例
    query = "G123平台是一个什么平台？"
    query_vector = zhipu_embedding([query])
    query_vector = np.array(query_vector)

    num_results = 3  # 返回前 3 个结果
    distances, indices = index.search(query_vector, num_results)

    print("Distances:", distances)
    print("Indices:", indices)

    for idx in indices[0]:
        print(f"Metadata for index {idx}: {metadata[idx]}")

if __name__ == '__main__':
    main()