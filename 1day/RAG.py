# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import re

import sys
import io

# 设置标准输出/错误流的编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 设置环境变量
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["OPENAI_API_ENCODING"] = "utf-8"

client = OpenAI(
    api_key="sk-29c3c3d895464df3b940b8d7f2896d5a",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    default_headers={"Content-Type": "application/json; charset=utf-8"}  # 关键修复
)



# ========================
# 1. 加载中文糖尿病数据集
# ========================
def load_chinese_diabetes_data():
    """加载中文糖尿病知识库"""
    # 方法1：从阿里云天池加载DiaKG糖尿病知识图谱（需申请）
    # 此处使用预处理好的样本数据（完整数据集需申请）
    data_url = "https://gist.githubusercontent.com/ai-med-china/raw/diabetes_kg_sample.csv"

    try:
        # 尝试在线加载
        med_data = pd.read_csv(data_url)
        print(f"成功加载在线数据集，共 {len(med_data)} 条记录")
    except:
        # 备用方案：本地模拟数据
        print("在线加载失败，使用本地模拟数据")
        med_data = pd.DataFrame([
            {"head": "二甲双胍", "relation": "适应症", "tail": "2型糖尿病",
             "text": "二甲双胍是治疗2型糖尿病的首选药物，特别适用于肥胖患者。"},
            {"head": "二甲双胍", "relation": "禁忌症", "tail": "肾功能不全",
             "text": "当患者eGFR<45ml/min/1.73m²时，禁用二甲双胍。"},
            {"head": "SGLT2抑制剂", "relation": "作用机制", "tail": "促进尿糖排泄",
             "text": "SGLT2抑制剂通过抑制肾脏对葡萄糖的重吸收，促进尿糖排泄来降低血糖。"},
            {"head": "GLP-1受体激动剂", "relation": "不良反应", "tail": "胃肠道反应",
             "text": "GLP-1受体激动剂常见不良反应包括恶心、呕吐等胃肠道症状。"},
            {"head": "胰岛素", "relation": "使用时机", "tail": "口服药失效",
             "text": "当口服降糖药效果不佳时，应考虑启动胰岛素治疗。"},
            {"head": "糖尿病", "relation": "并发症", "tail": "糖尿病肾病",
             "text": "糖尿病肾病是糖尿病常见的微血管并发症，表现为蛋白尿和肾功能下降。"},
            {"head": "HbA1c", "relation": "临床意义", "tail": "血糖控制指标",
             "text": "糖化血红蛋白(HbA1c)反映近2-3个月平均血糖水平，是糖尿病控制的重要指标。"},
            {"head": "糖尿病足", "relation": "预防措施", "tail": "足部检查",
             "text": "每日检查足部、保持足部清洁干燥是预防糖尿病足的关键措施。"},
            {"head": "低血糖", "relation": "处理", "tail": "15克葡萄糖",
             "text": "发生低血糖时，应立即补充15克葡萄糖或含糖食物。"},
            {"head": "糖尿病饮食", "relation": "原则", "tail": "控制总热量",
             "text": "糖尿病饮食治疗的核心原则是控制每日总热量摄入，均衡营养。"}
        ])

    # 创建完整文本字段
    med_data['full_text'] = med_data.apply(
        lambda x: f"{x['head']}{x['relation']}{x['tail']}。{x['text']}", axis=1
    )
    return med_data


# 加载数据
med_data = load_chinese_diabetes_data()
print("\n知识库示例:")
print(med_data[['head', 'relation', 'tail']].head(3))

# ===========================
# 2. 构建向量数据库
# ===========================
print("\n正在构建向量数据库...")

# 使用中文优化的文本嵌入模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 生成文本向量
texts = med_data['full_text'].tolist()
embeddings = model.encode(texts, convert_to_tensor=True)
embeddings_np = embeddings.cpu().numpy().astype('float32')

# 创建FAISS索引
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)
print(f"向量数据库构建完成! 维度: {dimension}, 记录数: {len(embeddings_np)}")


# ===========================
# 3. 检索模块
# ===========================
def retrieve_docs(question: str, top_k: int = 3):
    """检索相关医学知识"""
    # 问题向量化
    question_embedding = model.encode([question], convert_to_tensor=True)
    question_embedding_np = question_embedding.cpu().numpy().astype('float32')

    # 执行相似度搜索
    distances, indices = index.search(question_embedding_np, top_k)

    # 返回结果
    results = med_data.iloc[indices[0]].copy()
    results['similarity'] = 1 - distances[0] / 2  # 转换为相似度分数(0-1)
    return results


# 测试检索
test_question = "哪些糖尿病患者不能使用二甲双胍？"
retrieved = retrieve_docs(test_question)
print(f"\n问题: '{test_question}' 的检索结果:")
print(retrieved[['head', 'relation', 'tail', 'similarity']])


# ===========================
# 4. 增强生成模块
# ===========================
def generate_rag_answer(question: str):
    """生成RAG增强回答"""
    # 检索相关知识
    context_docs = retrieve_docs(question)

    # 构建上下文提示
    context_text = "相关医学知识:\n"
    for i, (_, row) in enumerate(context_docs.iterrows()):
        context_text += f"{i + 1}. {row['full_text']} (可信度: {row['similarity']:.2f})\n"

    # 创建增强提示词
    prompt = f"""您是一名专业的糖尿病医生，请根据以下权威知识回答问题：
        {context_text}
        
        问题：{question}
        
        回答要求：
        1. 基于提供的医学知识回答
        2. 如涉及禁忌症或警告，请用⚠️标识
        3. 引用相关知识点的序号作为参考
        4. 使用中文回答，语言简洁专业
        """

    # 调用生成模型
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # 降低随机性
        max_tokens=500
    )

    return response.choices[0].message.content


# ===========================
# 5. 测试与对比
# ===========================
def test_rag_system():
    """测试RAG系统"""
    questions = [
        "肾功能不全的糖尿病患者可以使用哪些降糖药？",
        "如何预防糖尿病足？",
        "二甲双胍的主要禁忌症是什么？",
        "HbA1c的正常范围是多少？"
    ]

    for q in questions:
        print(f"\n{'=' * 50}")
        print(f"问题: {q}")

        # 原始LLM回答
        raw_response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": q}]
        ).choices[0].message.content
        print(f"\n[原始LLM回答]:\n{raw_response}")

        # RAG增强回答
        rag_response = generate_rag_answer(q)
        print(f"\n[RAG增强回答]:\n{rag_response}")

        # 显示检索到的知识
        print("\n[检索到的知识]:")
        retrieved = retrieve_docs(q)
        for i, (_, row) in enumerate(retrieved.iterrows()):
            print(f"{i + 1}. {row['full_text']} (相似度: {row['similarity']:.2f})")


# 运行测试
print("\n正在测试RAG系统...")
test_rag_system()