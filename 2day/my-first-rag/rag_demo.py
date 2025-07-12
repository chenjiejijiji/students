import chardet
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


# 方法一：使用二进制模式读取并自动检测编码
def load_document_with_encoding(file_path):
    """安全加载文档，自动处理编码问题"""
    with open(file_path, 'rb') as f:  # 以二进制模式打开
        raw_data = f.read()

    # 检测文件编码
    encoding_detection = chardet.detect(raw_data)
    encoding = encoding_detection['encoding'] or 'utf-8'
    confidence = encoding_detection['confidence']

    print(f"检测到文件编码: {encoding} (置信度: {confidence:.2f})")

    # 尝试使用检测到的编码解码
    try:
        text = raw_data.decode(encoding)
    except UnicodeDecodeError:
        print(f"使用 {encoding} 解码失败，尝试回退到 latin-1")
        # latin-1 可以解码任何字节序列
        text = raw_data.decode('latin-1', errors='replace')

    # 创建文档对象
    return [Document(page_content=text, metadata={"source": file_path})]


# 方法二：如果方法一仍然失败，使用此替代方法
def safe_load_document(file_path):
    """更安全的文档加载方法"""
    try:
        # 尝试使用UTF-8加载
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        try:
            # 尝试使用GBK（常见于中文Windows）
            with open(file_path, 'r', encoding='gbk') as f:
                text = f.read()
        except UnicodeDecodeError:
            # 最后尝试使用latin-1（不会失败，但可能显示乱码）
            with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
                text = f.read()
            print("警告：使用latin-1编码加载，某些字符可能显示不正确")

    # 创建文档对象
    return [Document(page_content=text, metadata={"source": file_path})]

# 2. 加载文档
file_path = "./data/knowledge.txt"
# 选择一种方法加载文档
try:
    documents = load_document_with_encoding(file_path)
except Exception as e:
    print(f"自动编码检测失败: {e}")
    print("使用安全加载方法...")
    documents = safe_load_document(file_path)

print(f"加载了 {len(documents)} 个文档")

# 3. 切分文档 (Chunking)
# 关键步骤：将大文档切成适合检索的小片段
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 每个块的最大字符数 (根据你的文本调整)
    chunk_overlap=50,  # 块之间的重叠字符数 (有助于保持上下文)
    length_function=len,  # 计算长度的函数 (这里是字符数)
    add_start_index=True,  # 可选：在元数据中添加原始文档中的起始索引
)
chunks = text_splitter.split_documents(documents)
print(f"将文档切分成 {len(chunks)} 个块 (chunks)")
# 打印前2个块看看效果
for i, chunk in enumerate(chunks[:2]):
    print(f"\n**块 #{i+1}:**")
    print(chunk.page_content)
    print("-" * 50)

# 4. 设置嵌入模型 (Embedding Model)
# 使用 Sentence Transformers 的一个轻量级开源模型
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") # 这是一个广泛使用的、效果不错且速度较快的模型

# 5. 将块存储到向量数据库 (ChromaDB)
# Chroma 会自动处理向量化和索引创建
vector_db = Chroma.from_documents(
    documents=chunks,  # 我们切分好的文本块
    embedding=embedding_function,  # 我们选择的嵌入函数
    persist_directory="./chroma_db",  # 指定一个目录持久化存储数据库 (下次运行就不用重新生成了)
)
print(f"已将 {len(chunks)} 个块嵌入并存储到向量数据库 './chroma_db'")

# 6. 从已存在的向量数据库创建检索器 (Retriever)
# 即使重启脚本，只要 persist_directory 相同，就能加载之前构建的数据库
vector_db = Chroma(
    persist_directory="./chroma_db",  # 指向我们之前存储的数据库
    embedding_function=embedding_function,
)
# 创建检索器，配置其返回最相关的 K 个块
retriever = vector_db.as_retriever(search_kwargs={"k": 2})  # 尝试调整 k 值 (1, 2, 3...)

# 7. 设置生成器 (Generator) - 选择你的 LLM
# 创建原生 OpenAI 客户端
client  = OpenAI(
    api_key="sk-29c3c3d895464df3b940b8d7f2896d5a",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    default_headers={"Content-Type": "application/json; charset=utf-8"}  # 关键修复
)


# 方法一：修改自定义函数（支持额外参数）
def openai_invoke(prompt: str, **kwargs) -> str:
    """处理 LangChain 传递的额外参数"""
    # 确保提示是字符串
    if not isinstance(prompt, str):
        # 如果是字典，提取需要的部分
        if isinstance(prompt, dict):
            # 尝试提取常见键值
            prompt = prompt.get("input", "") or prompt.get("query", "") or str(prompt)
        else:
            prompt = str(prompt)

    # 处理可能的 stop 参数
    stop_sequences = kwargs.get("stop", None)

    # 调用 OpenAI API
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        stop=stop_sequences
    )
    return response.choices[0].message.content


# 创建 Runnable 对象
from langchain_core.runnables import RunnableLambda

llm_runnable = RunnableLambda(openai_invoke)



# 8. 组装 RAG 链
# 使用 LangChain 提供的简便 RetrievalQA 链
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_runnable,  # 我们选择的生成器模型
    chain_type="stuff",  # 简单地将所有检索到的上下文“塞”进提示词 (适合小k值)
    retriever=retriever,  # 我们配置的检索器
    return_source_documents=True,  # 非常重要！返回检索到的源文档用于调试
    verbose=False,  # 设置为 True 可以看到链执行的详细过程
)

# 9. 提问！
print("\n===== 开始提问吧！输入 'exit' 退出 =====")
while True:
    question = input("\n你的问题: ")
    if question.lower() == "exit":
        break

    # 调用 RAG 链获取答案
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    source_docs = result["source_documents"]

    # 打印答案
    print("\n**答案:**", answer)

    # 打印来源 (非常重要的调试信息！)
    print("\n**来源 (检索到的块):**")
    for i, doc in enumerate(source_docs):
        print(f"[来源 {i+1}]")
        print(doc.page_content)
        print("-" * 50)
