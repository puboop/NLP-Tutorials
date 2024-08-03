import numpy as np
import datetime
import os
import requests
import pandas as pd
import re
import itertools

PAD_ID = 0


class DateData:
    def __init__(self, n):
        np.random.seed(1)
        self.date_cn = []
        self.date_en = []
        # 随机生成时间戳
        for timestamp in np.random.randint(143835585, 2043835585, n):
            date = datetime.datetime.fromtimestamp(timestamp)
            self.date_cn.append(date.strftime("%y-%m-%d"))  # 年（后两位），月，日
            self.date_en.append(date.strftime("%d/%b/%Y"))  # 日，月（英文），年（整个年份）
        self.vocab = set(
            [str(i) for i in range(0, 10)]
            + ["-", "/", "<GO>", "<EOS>"]  # 特殊字符，以及开始标识，结束标识
            + [i.split("/")[1] for i in self.date_en]  # 得到所有的月（英文）
        )  # 整个特征为，0-9（字符串格式），生成的相应的月份（英文）
        # 生成相应的key对应的索引值（也就是下标值）
        self.v2i = {v: i for i, v in enumerate(sorted(list(self.vocab)), start=1)}
        self.v2i["<PAD>"] = PAD_ID
        self.vocab.add("<PAD>")
        # 生成值对应key，也就是下标值对应key 用于将数字转换为字符
        self.i2v = {i: v for v, i in self.v2i.items()}
        # x为所有的时间key对应的索引
        # y为每个时间的开头部分以及结尾部分
        self.x, self.y = [], []
        for cn, en in zip(self.date_cn, self.date_en):
            self.x.append([self.v2i[v] for v in cn])  # 将每个时间key对应的value获取出来
            self.y.append([self.v2i["<GO>"], ]  # 开始标识
                          + [self.v2i[v] for v in en[:3]]  # 获取日期以及/
                          + [self.v2i[en[3:6]], ]  # 获取月份
                          + [self.v2i[v] for v in en[6:]]  # 获取年份
                          + [self.v2i["<EOS>"], ]  # 结束标识
                          )
        self.x, self.y = np.array(self.x), np.array(self.y)
        self.start_token = self.v2i["<GO>"]
        self.end_token = self.v2i["<EOS>"]

    def sample(self, n=64):
        # 随机获取从0到整个时间索引的长度的序列，大小为n
        bi = np.random.randint(0, len(self.x), size=n)
        # 取出x对应的索引，以及y对应的索引
        bx, by = self.x[bi], self.y[bi]
        # 获取到解码长度
        decoder_len = np.full((len(bx),), by.shape[1] - 1, dtype=np.int32)
        return bx, by, decoder_len

    def idx2str(self, idx):
        """将预测结果转为字符串形式"""
        x = []
        for i in idx:
            x.append(self.i2v[i])
            if i == self.end_token:
                break
        return "".join(x)

    @property
    def num_word(self):
        # 获取整个特征的个数
        return len(self.vocab)


def pad_zero(seqs, max_len):
    """
    用于将一组序列进行填充，使它们的长度统一为指定的最大长度 max_len。
    填充的方式是在序列的末尾补上特定的填充值 PAD_ID。这是在自然语言处理和机器学习中处理变长序列时常用的预处理方法
    """
    padded = np.full((len(seqs), max_len), fill_value=PAD_ID, dtype=np.long)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded


def maybe_download_mrpc(save_dir="./MRPC/", proxy=None):
    train_url = 'https://mofanpy.com/static/files/MRPC/msr_paraphrase_train.txt'
    test_url = 'https://mofanpy.com/static/files/MRPC/msr_paraphrase_test.txt'
    os.makedirs(save_dir, exist_ok=True)
    proxies = {"http": proxy, "https": proxy}
    for url in [train_url, test_url]:
        raw_path = os.path.join(save_dir, url.split("/")[-1])
        if not os.path.isfile(raw_path):
            print("downloading from %s" % url)
            r = requests.get(url, proxies=proxies)
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(r.text.replace('"', "<QUOTE>"))
                print("completed")


def _text_standardize(text):
    text = re.sub(r'—', '-', text)
    text = re.sub(r'–', '-', text)
    text = re.sub(r'―', '-', text)
    text = re.sub(r" \d+(,\d+)?(\.\d+)? ", " <NUM> ", text)
    text = re.sub(r" \d+-+?\d*", " <NUM>-", text)
    return text.strip()


def _process_mrpc(dir="./MRPC", rows=None):
    data = {"train": None, "test": None}
    files = os.listdir(dir)
    for f in files:
        df = pd.read_csv(os.path.join(dir, f), sep='\t', nrows=rows)
        k = "train" if "train" in f else "test"
        data[k] = {"is_same": df.iloc[:, 0].values, "s1": df["#1 String"].values, "s2": df["#2 String"].values}
    vocab = set()
    # 将所有字符进行唯一性处理
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            # 取出训练模式下的相应数据
            for i in range(len(data[n][m])):
                # 字符串处理
                data[n][m][i] = _text_standardize(data[n][m][i].lower())
                cs = data[n][m][i].split(" ")
                vocab.update(set(cs))
    # 对每个词进行词索引处理
    v2i = {v: i for i, v in enumerate(sorted(vocab), start=1)}
    v2i["<PAD>"] = PAD_ID
    v2i["<MASK>"] = len(v2i)
    v2i["<SEP>"] = len(v2i)
    v2i["<GO>"] = len(v2i)
    i2v = {i: v for v, i in v2i.items()}
    for n in ["train", "test"]:
        for m in ["s1", "s2"]:
            # 词索引id
            data[n][m + "id"] = [[v2i[v] for v in c.split(" ")] for c in data[n][m]]
    return data, v2i, i2v


class MRPCData:
    num_seg = 3
    pad_id = PAD_ID

    def __init__(self, data_dir="./MRPC/", rows=None, proxy=None):
        maybe_download_mrpc(save_dir=data_dir, proxy=proxy)
        # data: 包含训练和测试数据。
        # self.v2i: 词汇表，词到索引的映射。
        # self.i2v: 词汇表，索引到词的映射。
        data, self.v2i, self.i2v = _process_mrpc(data_dir, rows)
        # 计算最大长度
        # 计算训练和测试数据中序列的最大长度，考虑了句子分隔符（<SEP>）和开始标记（<GO>）。
        # len(s1): 第一个句子的长度。
        # len(s2): 第二个句子的长度。
        # + 3: 加上<GO>和两个<SEP>标记。
        self.max_len = max([len(s1) + len(s2) + 3
                            for s1, s2 in zip(data["train"]["s1id"] + data["test"]["s1id"],
                                              data["train"]["s2id"] + data["test"]["s2id"])])

        # 计算每对句子的长度。
        # data["train"]["s1id"]: 训练数据中的第一个句子ID序列。
        # data["train"]["s2id"]: 训练数据中的第二个句子ID序列。
        self.xlen = np.array([[len(data["train"]["s1id"][i]), len(data["train"]["s2id"][i])]
                              for i in range(len(data["train"]["s1id"]))], dtype=int)
        # 计算每段话的长度
        # 为每对句子生成输入序列，添加特殊标记<GO>和<SEP>。
        x = [[self.v2i["<GO>"]]
             + data["train"]["s1id"][i]
             + [self.v2i["<SEP>"]]
             + data["train"]["s2id"][i]
             + [self.v2i["<SEP>"]]
             for i in range(len(self.xlen))]
        # 长度填充
        self.x = pad_zero(x, max_len=self.max_len)
        # self.nsp_y: 对应的标签，是否为同义句。
        self.nsp_y = data["train"]["is_same"][:, None]

        # 填充形状为self.x.shape 填充数为self.num_seg-1
        # self.seg: 段信息，指定每个词的段（句子）编号。
        # self.num_seg: 段的数量。
        # si: 第一个句子的结束位置。
        # si_: 第二个句子的结束位置。
        self.seg = np.full(self.x.shape, self.num_seg - 1, np.int32)
        for i in range(len(x)):
            si = self.xlen[i][0] + 2
            self.seg[i, :si] = 0
            si_ = si + self.xlen[i][1] + 1
            self.seg[i, si:si_] = 1
        # 生成词ID的集合，排除填充标记<PAD>、掩码标记<MASK>和分隔标记<SEP>。
        self.word_ids = np.array(list(
            set(self.i2v.keys()).difference([self.v2i[v] for v in ["<PAD>", "<MASK>", "<SEP>"]])
        ))

    def sample(self, n):
        bi = np.random.randint(0, self.x.shape[0], size=n)
        bx, bs, bl, by = self.x[bi], self.seg[bi], self.xlen[bi], self.nsp_y[bi]
        return bx, bs, bl, by

    @property
    def num_word(self):
        return len(self.v2i)

    @property
    def mask_id(self):
        return self.v2i["<MASK>"]


class MRPCSingle:
    pad_id = PAD_ID

    def __init__(self, data_dir="./MRPC/", rows=None, proxy=None):
        maybe_download_mrpc(save_dir=data_dir, proxy=proxy)
        data, self.v2i, self.i2v = _process_mrpc(data_dir, rows)
        # 最大长度计算
        self.max_len = max([len(s) + 2 for s in data["train"]["s1id"] + data["train"]["s2id"]])
        # 取出每个词的词索引
        x = [[self.v2i["<GO>"]] + data["train"]["s1id"][i] + [self.v2i["<SEP>"]]
             for i in range(len(data["train"]["s1id"]))]
        x += [[self.v2i["<GO>"]] + data["train"]["s2id"][i] + [self.v2i["<SEP>"]]
              for i in range(len(data["train"]["s2id"]))]
        # 不足的填充0
        self.x = pad_zero(x, max_len=self.max_len)
        self.word_ids = np.array(list(set(self.i2v.keys()).difference([self.v2i["<PAD>"]])))

    def sample(self, n):
        bi = np.random.randint(0, self.x.shape[0], size=n)
        bx = self.x[bi]
        return bx

    @property
    def num_word(self):
        return len(self.v2i)


class Dataset:
    def __init__(self, x, y, v2i, i2v):
        """
        x: 所有的词
        y: 词对应的次数
        v2i: 词对应的次数 字典
        i2v: 次数对应的词 字典
        """
        self.x, self.y = x, y
        self.v2i, self.i2v = v2i, i2v
        self.vocab = v2i.keys()

    def sample(self, n):
        # 生成0-len(self.x)范围的数，维度为n
        b_idx = np.random.randint(0, len(self.x), n)
        bx, by = self.x[b_idx], self.y[b_idx]
        return bx, by

    @property
    def num_word(self):
        # 总共有多少词
        return len(self.v2i)


def process_w2v_data(corpus, skip_window=2, method="skip_gram"):
    all_words = [sentence.split(" ") for sentence in corpus]
    # itertools.chain将 docs_words 中的所有单词列表展开为一个单一的迭代器。
    all_words = np.array(list(itertools.chain(*all_words)))
    # vocab sort by decreasing frequency for the negative sampling below (nce_loss).
    # vocab 每个唯一的词
    # v_count 计算词汇表 vocab 和每个词的出现次数 v_count。
    vocab, v_count = np.unique(all_words, return_counts=True)
    vocab = vocab[np.argsort(v_count)[::-1]]

    print("all vocabularies sorted from more frequent to less frequent:\n", vocab)
    v2i = {v: i for i, v in enumerate(vocab)}
    i2v = {i: v for v, i in v2i.items()}

    # pair data
    # 存储训练对
    pairs = []
    # 包含窗口范围内的所有相对位置（不包括0）
    js = [i for i in range(-skip_window, skip_window + 1) if i != 0]
    # 循环每个文档
    for c in corpus:
        words = c.split(" ")
        # 循环每个词出现的次数
        """
        v2i为所有单词的所出现的次数，为一个字典
        words为当前文章中所有的单词
        w_idx为当前文章中的单词在总文章中的单词中所出现的次数
        """
        w_idx = [v2i[w] for w in words]
        # Skip-Gram遍历每个单词 w_idx[i]，在窗口范围内生成 (中心词, 上下文词) 对。
        if method == "skip_gram":
            # 循环当前文章的所有单词数量
            for i in range(len(w_idx)):
                # 获取每个词的上下文词的索引
                for j in js:
                    if (i + j < 0) or (i + j >= len(w_idx)):
                        continue
                    # w_idx[i]当前词
                    # w_idx[i + j]当前词的上下文词
                    pairs.append((w_idx[i], w_idx[i + j]))  # (center, context) or (feature, target)
        # CBOW遍历每个单词 w_idx[i]，在窗口范围内生成 (上下文词列表, 中心词) 对
        elif method.lower() == "cbow":
            # 通过上下文从skip_window开始计算上下文词，截至到 len(w_idx) - skip_window来计算中心词的个数
            for i in range(skip_window, len(w_idx) - skip_window):
                context = []
                # 得到每个词的上下文
                for j in js:
                    # 计算每个词之前的词与之后的词
                    context.append(w_idx[i + j])
                # 将每个词的上下文放到全局上下文当中去
                pairs.append(context + [w_idx[i]])  # (contexts, center) or (feature, target)
        else:
            raise ValueError
    pairs = np.array(pairs)
    print("5 example pairs:\n", pairs[:5])
    # skip_gram 方法，将 pairs 拆分为输入 x 和输出 y，其中 x 是中心词，y 是上下文词
    if method.lower() == "skip_gram":
        x, y = pairs[:, 0], pairs[:, 1]
    # cbow 方法，将 pairs 拆分为输入 x 和输出 y，其中 x 是上下文词列表，y 是中心词
    elif method.lower() == "cbow":
        x, y = pairs[:, :-1], pairs[:, -1]
    else:
        raise ValueError
    # Dataset 对象，包含输入 x、输出 y、词汇到索引的映射 v2i 和索引到词汇的映射 i2v
    return Dataset(x, y, v2i, i2v)


def set_soft_gpu(soft_gpu):
    import tensorflow as tf
    if soft_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
