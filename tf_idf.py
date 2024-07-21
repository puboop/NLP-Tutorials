import numpy as np
from collections import Counter
import itertools
from visual import show_tfidf   # this refers to visual.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)

docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

docs_words = [d.replace(",", "").split(" ") for d in docs]
# 将 docs_words 中的所有单词列表展开为一个单一的迭代器。
# set将展开的单词迭代器转换为一个集合（set），从而得到所有不重复的单词（即词汇表 vocab）。
vocab = set(itertools.chain(*docs_words))
v2i = {v: i for i, v in enumerate(vocab)}
i2v = {i: v for v, i in v2i.items()}


def safe_log(x):
    mask = x != 0
    x[mask] = np.log(x[mask])
    return x


tf_methods = {
        "log": lambda x: np.log(1+x),
        "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
        "boolean": lambda x: np.minimum(x, 1),
        "log_avg": lambda x: (1 + safe_log(x)) / (1 + safe_log(np.mean(x, axis=1, keepdims=True))),
    }
idf_methods = {
        "log": lambda x: 1 + np.log(len(docs) / (x+1)),
        "prob": lambda x: np.maximum(0, np.log((len(docs) - x) / (x+1))),
        "len_norm": lambda x: x / (np.sum(np.square(x))+1),
    }


def get_tf(method="log"):
    """
    tf简称为词频：
    词频的计算：该词在当前文档中出现的次数除以该文档中所有的词的个数
    """
    """
    这里得到这么len(vocab), len(docs)行与列的DataFrame表格，
    vocab表示所有文档中的所有单词（去重后的单词，不包含标点符号）的总个数，
    docs表示总文档数
    """
    # term frequency: how frequent a word appears in a doc
    _tf = np.zeros((len(vocab), len(docs)), dtype=np.float64)    # [n_vocab, n_doc]
    for i, d in enumerate(docs_words):
        # Counter计算单个文档中的每个单词出现的总次数
        # 单词为键，出现的次数为值返回个类似字典的结构
        counter = Counter(d)
        for v in counter.keys():
            #### 计算词频 ####
            # counter[v]在文档中出现的次数
            # counter.most_common(1)[0][1]获取出现次数最多的词的次数
            # _tf[v2i[v], i]根据DataFrame表中的位置修改相应的值
            _tf[v2i[v], i] = counter[v] / counter.most_common(1)[0][1]
    # 从 tf_methods 字典中获取指定的词频加权方法。如果未找到该方法，则抛出 ValueError 异常。
    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    # 用获取的加权方法对 _tf 矩阵进行加权，并返回加权后的词频矩阵。
    return weighted_tf(_tf)


def get_idf(method="log"):
    """
    idf逆文档频率计算：
        该词在所有文档中出现的次数除总数次，取对数值
    """
    # inverse document frequency: low idf for a word appears in more docs, mean less important
    df = np.zeros((len(i2v), 1))
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
        df[i, 0] = d_count

    idf_fn = idf_methods.get(method, None)
    if idf_fn is None:
        raise ValueError
    return idf_fn(df)


def cosine_similarity(q, _tf_idf):
    # 计算查询向量 q 与文档集 TF-IDF 矩阵 _tf_idf 的余弦相似度。
    # 余弦相似度是一种常用的衡量两个向量之间相似度的方法。

    # 归一化查询向量
    """
    np.square(q)：计算查询向量 q 中每个元素的平方。
    np.sum(..., axis=0, keepdims=True)：沿着列方向对所有平方值求和，保留其二维形状。
    np.sqrt(...)：对上述和取平方根。
    q / ...：将查询向量 q 的每个元素除以其模长（即归一化）。
    unit_q 是归一化后的查询向量，其模长为 1。
    """
    unit_q = q / np.sqrt(np.sum(np.square(q), axis=0, keepdims=True))
    # 归一化文档集向量
    """
    类似地，计算文档集 TF-IDF 矩阵 _tf_idf 的每个列向量的模长，并对每个列向量进行归一化。
    unit_ds 是归一化后的文档集矩阵，每个列向量的模长为 1
    """
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    # 计算余弦相似度
    """
    unit_ds.T：将归一化后的文档集矩阵转置，使其形状变为 [n_docs, n_vocab]。
    unit_ds.T.dot(unit_q)：计算归一化后的查询向量与每个归一化后的文档向量之间的点积，得到一个形状为 [n_docs, 1] 的相似度向量。
    .ravel()：将相似度向量展平为一维数组
    """
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity


def docs_score(q, len_norm=False):
    # 用于计算查询 q 与文档集 docs 中每个文档的相似度分数。
    # 相似度通过余弦相似度计算，并可选择进行长度归一化
    q_words = q.replace(",", "").split(" ")

    # add unknown words
    # 初始化一个计数器 unknown_v 为0，用于统计查询中未见过的单词数量
    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:
            v2i[v] = len(v2i)
            i2v[len(v2i)-1] = v
            unknown_v += 1
    # 如果有未知单词（unknown_v > 0），则扩展 idf 和 tf_idf 矩阵，以容纳这些新单词
    # 通过将零矩阵与原始矩阵连接来实现扩展。
    if unknown_v > 0:
        # 将新搜索的当作一篇文档追加到末尾
        _idf = np.concatenate((idf, np.zeros((unknown_v, 1), dtype=np.float)), axis=0)
        _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_v, tf_idf.shape[1]), dtype=np.float)), axis=0)
    else:
        _idf, _tf_idf = idf, tf_idf
    # 计算tf
    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype=np.float)     # [n_vocab, 1]
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]
    """
    新的tf_idf = 旧的idf * 新的tf
    """
    q_vec = q_tf * _idf            # [n_vocab, 1]
    """
    将新的tf_idf 与 旧的tf_idf进行cosine进行计算相似度
    """
    q_scores = cosine_similarity(q_vec, _tf_idf)
    # 进行长度归一化。计算每个文档的长度，并将相似度分数 q_scores 除以文档长度。
    if len_norm:
        len_docs = [len(d) for d in docs_words]
        q_scores = q_scores / np.array(len_docs)
    return q_scores


def get_keywords(n=2):
    # 用于从文档集中提取每个文档的前 n 个关键词
    for c in range(3):
        col = tf_idf[:, c]
        # 使用 np.argsort(col) 对第 c 个文档的 TF-IDF 值进行排序，返回的是排序后的索引数组。
        # col 是一个包含第 c 个文档中每个词的 TF-IDF 值的数组。
        idx = np.argsort(col)[-n:]
        print("doc{}, top{} keywords {}".format(c, n, [i2v[i] for i in idx]))


tf = get_tf()           # [n_vocab, n_doc]
idf = get_idf()         # [n_vocab, 1]
tf_idf = tf * idf       # [n_vocab, n_doc]
print("tf shape(vecb in each docs): ", tf.shape)
print("\ntf samples:\n", tf[:2])
print("\nidf shape(vecb in all docs): ", idf.shape)
print("\nidf samples:\n", idf[:2])
print("\ntf_idf shape: ", tf_idf.shape)
print("\ntf_idf sample:\n", tf_idf[:2])


# test
get_keywords()
q = "I get a coffee cup"
scores = docs_score(q)
d_ids = scores.argsort()[-3:][::-1]
print("\ntop 3 docs for '{}':\n{}".format(q, [docs[i] for i in d_ids]))

# show_tfidf(tf_idf.T, [i2v[i] for i in range(tf_idf.shape[0])], "tfidf_matrix")