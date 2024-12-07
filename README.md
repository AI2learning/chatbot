# 智能客服

## Demo1
基于Dify进行快速搭建demo，适合并发量小，知识库不大的企业应用。

### demo1链接
https://udify.app/chat/bh8utMeQs0rdteEc

## Demo2

- pdf解析，后续涉及到表格或者图片可以考虑使用MinerU
- 文本切chunk，后续是一个调优的点
- 文本chunk进行向量化，本demo使用的智谱embedding3 api接口，维度2048，对于量不大的话，直接使用api接口，快速和性价比高。
- 存储vectors，对于数据量小和cpu环境，可以考虑使用faiss，对于数据量大和并发高，可以考虑后续使用milvus和pgvector等向量数据库。
- 意图识别，目前配的提示词，后续可以考虑微调小模型进行快速意图识别，当业务场景意图比较多的时候，以及需要调用不同的工具。
- 融合召回的领域知识，给大模型进行客户问题回答。

### demo2链接
http://106.14.45.210:6006/

### 代码说明
- `pdf_vectors.py`构建知识库。
- `app.py`进行stremlit多轮对话的构建。
