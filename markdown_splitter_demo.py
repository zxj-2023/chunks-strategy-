from langchain.text_splitter import MarkdownHeaderTextSplitter
import os

# 1. 定义要分割的标题级别及其对应的元数据键名
# 格式: (标题标记, 元数据键名)
headers_to_split_on = [
    ("#", "Header 1"),    # 一级标题，对应元数据键为 "Header 1"
    ("##", "Header 2"),   # 二级标题，对应元数据键为 "Header 2"
    ("###", "Header 3"),  # 三级标题，对应元数据键为 "Header 3"
]

# 2. 创建 MarkdownHeaderTextSplitter 实例
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 3. 读取 demo1.md 文件内容
with open("/workspace/demo1.md", "r", encoding="utf-8") as f:
    markdown_document = f.read()

# 4. 使用 MarkdownHeaderTextSplitter 分割文档
md_header_splits = markdown_splitter.split_text(markdown_document)

# 5. 打印分割结果概览
print(f"文档被分成了 {len(md_header_splits)} 个部分\n")

# 6. 打印前3个分块的详细信息
print("前3个文档块的详细信息:")
for i, split in enumerate(md_header_splits[:3]):
    print(f"块 {i+1}:")
    print(f"元数据: {split.metadata}")
    # 打印内容的前80个字符作为预览
    content_preview = split.page_content[:80] + "..." if len(split.page_content) > 80 else split.page_content
    print(f"内容预览: {content_preview}")
    print("-" * 50)


# 7. 结合 RecursiveCharacterTextSplitter 限制块长度
print("\n======== 结合 RecursiveCharacterTextSplitter 限制块长度 ========\n")
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 创建 RecursiveCharacterTextSplitter 实例，设置块大小和重叠
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

# 对已经按标题分割的块进行进一步的细粒度分割
all_splits = []
for header_split in md_header_splits:
    # 对每个块应用 RecursiveCharacterTextSplitter
    splits = child_splitter.split_documents([header_split])
    all_splits.extend(splits)

print(f"结合 RecursiveCharacterTextSplitter 后，文档被分成了 {len(all_splits)} 个更小的部分。")

# 打印一个细分后块的示例，以展示其元数据和内容
print("\n一个细分后块的示例:")
example_split = all_splits[5]  # 选择一个示例块
print(f"元数据: {example_split.metadata}")
print(f"内容: \n{example_split.page_content}")
print(f"\n这个块的长度是: {len(example_split.page_content)}个字符")
print("-" * 50)
