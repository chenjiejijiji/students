# 首先安装必要的库
# pip install sentence-transformers

# 医学术语映射表
medical_synonyms = {
    "肾功能不全": ["eGFR降低", "肌酐升高", "肾功能异常", "CKD"],
    "二甲双胍": ["格华止", "Metformin", "二甲双胍缓释片"]
}

def expand_query(query):
    """扩展查询术语"""
    expanded = []
    for word in query.split():
        if word in medical_synonyms:
            expanded += medical_synonyms[word]
        expanded.append(word)
    return " ".join(expanded)

# 导入所需模块
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 1. 初始化模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. 计算两个医学术语的相似度
term1 = "二甲双胍"
term2 = "盐酸二甲双胍片"

# 将文本转换为向量
vec1 = model.encode(term1, convert_to_tensor=True)
vec2 = model.encode(term2, convert_to_tensor=True)

# 计算余弦相似度
similarity_score = util.cos_sim(vec1, vec2).item()
print(f"'{term1}' 和 '{term2}' 的语义相似度: {similarity_score:.4f}")

# 3. 医学知识检索系统
knowledge_base = [
    "SGLT2抑制剂通过促进尿糖排泄降低血糖",
    "二甲双胍是2型糖尿病一线用药",
    "eGFR<45时禁用二甲双胍",
    "GLP-1受体激动剂可延缓胃排空",
    "胰岛素适用于口服降糖药失效的患者"
]
# 增强后的知识库
enhanced_kb = [
    "药物作用机制: SGLT2抑制剂通过促进尿糖排泄降低血糖",
    "临床应用: 二甲双胍是2型糖尿病一线用药",
    "禁忌症警告: 当患者肾功能不全(eGFR<45ml/min)时禁用二甲双胍",  # 增强条目
    "药物副作用: GLP-1受体激动剂可延缓胃排空",
    "治疗策略: 胰岛素适用于口服降糖药失效的患者"
]

question = "肾功能不全患者能用二甲双胍吗？"

# 扩展后的问题
original_question = "肾功能不全患者能用二甲双胍吗？"
expanded_question = expand_query(original_question)
print(f"扩展后问题: {expanded_question}")

# 将知识库转换为向量
kb_embeddings = model.encode(enhanced_kb, convert_to_tensor=True)

# 将问题转换为向量
q_embedding = model.encode(expanded_question, convert_to_tensor=True)

# 计算问题与知识库中每个条目的相似度
scores = util.cos_sim(q_embedding, kb_embeddings)[0]

# 获取最相关的结果
max_score_index = scores.argmax().item()
max_score = scores[max_score_index].item()
most_relevant = knowledge_base[max_score_index]

print(f"\n问题: '{question}'")
print(f"最相关知识: '{most_relevant}' (相似度: {max_score:.4f})")

# 4. 查看所有知识条目的相似度排序
print("\n知识库条目相似度排序:")
sorted_indices = np.argsort(-scores.cpu().numpy())  # 降序排列
for idx in sorted_indices:
    print(f"{knowledge_base[idx]} - 相似度: {scores[idx]:.4f}")