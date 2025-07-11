from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import DashScopeEmbeddings
import numpy as np
import re

with open('dream.txt', 'r', encoding='utf-8') as f:
    essay = f.read()

embeddings_model = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key="sk-ba2dda3817f145d7af141fdf32e31d90",

    )

semantic_chunk=SemanticChunker(
    embeddings=embeddings_model,#嵌入模型
    breakpoint_threshold_type="percentile",#定义如何计算语义断点阈值
    breakpoint_threshold_amount=95,#设定阈值
    #min_chunk_size=500#限制生成块最小的字符数，避免生成无意义的块
    sentence_split_regex=r'[。！？.\n]',#语句切分
)

# 分块
docs=semantic_chunk.create_documents([essay])
print(len(docs))
# distances, sentences = semantic_chunk._calculate_sentence_distances([essay])
# print("Sentences found:", len(sentences))

# breakpoint_percentile_threshold = 95
# breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
# print(breakpoint_distance_threshold)

print(f"拆分块数量：{len(docs)}")
for i,doc in enumerate(docs):
    print(f"第{i}个分块的字符长度：{len(doc.page_content)}")
    print("分块内容：", doc.page_content)
    print("\n")
