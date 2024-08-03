# [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
import tensorflow as tf
from tensorflow import keras
import utils  # this refers to utils.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)
import time
from transformer import Encoder
import pickle
import os


class GPT(keras.Model):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, lr, max_seg=3, drop_rate=0.1, padding_idx=0):
        """
        model_dim: 模型的隐藏层维度，定义嵌入层和其他层的输出维度
        max_len: 输入序列的最大长度
        n_layer: 编码器的层数
        n_head: 多头注意力机制中的头数
        n_vocab: 词汇表的大小
        lr: 学习率，用于Adam优化器
        max_seg: 最大段数，用于段嵌入
        drop_rate: Dropout层的丢弃率
        padding_idx: 用于表示填充的索引值，通常用于忽略序列中的填充部分
        """
        super().__init__()
        self.padding_idx = padding_idx
        self.n_vocab = n_vocab
        self.max_len = max_len

        # I think task emb is not necessary for pretraining,
        # because the aim of all tasks is to train a universal sentence embedding
        # the body encoder is the same across all tasks,
        # and different output layer defines different task just like transfer learning.
        # finetuning replaces output layer and leaves the body encoder unchanged.

        # 我认为emb任务不是预训练所必需的，
        # 因为所有任务的目标都是训练一个通用的句子嵌入
        # 身体编码器在所有任务中都是相同的，
        # 不同的输出层定义了不同的任务，就像迁移学习一样
        # 微调替换了输出层，并保持主体编码器不变

        # self.task_emb = keras.layers.Embedding(
        #     input_dim=n_task, output_dim=model_dim,  # [n_task, dim]
        #     embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        # )

        # word_emb: 词嵌入层，将词汇表中的每个词索引映射到一个向量
        self.word_emb = keras.layers.Embedding(
            input_dim=n_vocab, output_dim=model_dim,  # [n_vocab, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )
        # segment_emb: 段嵌入层，用于表示不同的句子段落
        self.segment_emb = keras.layers.Embedding(
            input_dim=max_seg, output_dim=model_dim,  # [max_seg, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )
        # position_emb: 位置编码，用于表示序列中不同位置的词
        self.position_emb = self.add_weight(
            name="pos", shape=[1, max_len, model_dim], dtype=tf.float32,  # [1, step, dim]
            initializer=keras.initializers.RandomNormal(0., 0.01))

        # encoder: 编码器类实例，包含多头自注意力机制和前馈神经网络
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        # task_mlm: 全连接层，用于Masked Language Model任务的输出
        self.task_mlm = keras.layers.Dense(n_vocab)
        # task_nsp: 全连接层，用于Next Sentence Prediction任务的输出
        self.task_nsp = keras.layers.Dense(2)
        # cross_entropy: 使用稀疏类别交叉熵作为损失函数，from_logits=True表示输入未经过softmax激活，reduction="none"表示不聚合损失值，通常用于序列模型中计算每个时间步的损失
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.opt = keras.optimizers.Adam(lr)

    def call(self, seqs, segs, training=False):
        """
        seqs: 输入序列，通常是词索引的列表或张量
        segs: 段落信息，表示输入序列中的每个词属于哪个段落
        training: 布尔值，指示当前模式是训练模式还是推理模式
        """
        # self.input_emb: 通过调用嵌入层（可能包括词嵌入、段嵌入和位置编码），将输入序列和段落信息转换为嵌入向量
        # embed: 嵌入后的向量表示，形状为[n, step, dim]，其中n是批量大小，step是序列长度，dim是嵌入维度
        embed = self.input_emb(seqs, segs)  # [n, step, dim]
        # self.encoder: 编码器模块，将嵌入向量输入到编码器中进行处理
        # training: 表示当前是否处于训练模式，如果是训练模式，则可能会应用dropout等正则化技术
        # mask: 掩码，忽略输入序列中的填充部分，确保填充不会影响模型的计算
        # z: 编码器的输出，形状为[n, step, dim]，与输入嵌入向量形状相同
        z = self.encoder(embed, training=training, mask=self.mask(seqs))  # [n, step, dim]
        # self.task_mlm: 全连接层，用于计算每个时间步上词汇表中每个词的概率分布
        # mlm_logits: MLM任务的输出，形状为[n, step, n_vocab]，表示每个时间步上词汇表中每个词的logits
        mlm_logits = self.task_mlm(z)  # [n, step, n_vocab]
        nsp_logits = self.task_nsp(tf.reshape(z, [z.shape[0], -1]))  # [n, n_cls]
        return mlm_logits, nsp_logits

    def step(self, seqs, segs, seqs_, nsp_labels):
        with tf.GradientTape() as tape:
            # 前向传播计算模型的预测输出
            mlm_logits, nsp_logits = self.call(seqs, segs, training=True)
            # 创建填充掩码（padding mask），用于标记输入序列中不是填充部分的位置
            pad_mask = tf.math.not_equal(seqs_, self.padding_idx)
            # self.cross_entropy(seqs_, mlm_logits): 计算预测 logits 和真实标签之间的交叉熵损失
            # tf.reduce_mean(...): 计算损失的平均值
            pred_loss = tf.reduce_mean(tf.boolean_mask(self.cross_entropy(seqs_, mlm_logits), pad_mask))
            # tf.boolean_mask(..., pad_mask): 使用填充掩码过滤掉填充部分的损失
            # 计算 NSP 预测 logits 和真实标签之间的交叉熵损失。
            nsp_loss = tf.reduce_mean(self.cross_entropy(nsp_labels, nsp_logits))
            # 总损失 = MLM 损失 + NSP 损失的加权和
            # 0.2: NSP 损失的权重系数，可以根据实际情况调整
            loss = pred_loss + 0.2 * nsp_loss
            grads = tape.gradient(loss, self.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, mlm_logits

    def input_emb(self, seqs, segs):
        """
        seqs: 输入序列，通常是词索引的列表或张量，形状为[n, step]
        segs: 段落信息，表示输入序列中的每个词属于哪个段落，形状为[n, step]

        self.word_emb(seqs): 词嵌入层，将输入序列中的词索引转换为词向量，形状为[n, step, dim]
        self.segment_emb(segs): 段嵌入层，将段落信息转换为段向量，形状为[n, step, dim]
        self.position_emb: 位置嵌入矩阵，形状为[1, max_len, dim]，用于为输入序列中的每个位置添加位置编码由于位置嵌入在整个序列中是相同的，因此它不依赖于输入的seqs和segs，可以直接使用预定义的嵌入矩阵
        """
        return self.word_emb(seqs) + self.segment_emb(segs) + self.position_emb  # [n, step, dim]

    def mask(self, seqs):
        """
        生成掩码矩阵的方法，用于在序列处理过程中强制执行注意力机制的约束，使得当前位置只能看到前面的位置信息，避免看到后续的位置信息

         abcd--
        a011111
        b001111
        c000111
        d000011
        -000011
        -000011

        force head not to see afterward. eg.
        a is a embedding for a---
        b is a embedding for ab--
        c is a embedding for abc-
        later, b embedding will + b another embedding from previous residual input to predict c
        """

        # tf.ones((self.max_len, self.max_len)): 创建一个全是1的矩阵，形状为[max_len, max_len]
        # tf.linalg.band_part: 生成上三角矩阵，并减去1得到下三角掩码矩阵
        """三角矩阵示意图
        [[0, 1, 1, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]]
        """
        mask = 1 - tf.linalg.band_part(tf.ones((self.max_len, self.max_len)), -1, 0)
        # 处理填充符号
        # tf.math.equal(seqs, self.padding_idx): 找到序列中等于padding_idx的位置，返回一个布尔矩阵，形状为[batch_size, seq_len]
        # tf.where: 用于将填充位置的掩码设置为1，以确保这些位置不参与计算
        pad = tf.math.equal(seqs, self.padding_idx)
        mask = tf.where(pad[:, tf.newaxis, tf.newaxis, :], 1, mask[tf.newaxis, tf.newaxis, :, :])
        # 掩码机制: 强制注意力机制只能关注到当前位置及其之前的位置，防止模型在预测下一个词时看到未来的信息
        # 处理填充符号: 确保填充位置不参与计算，从而避免模型处理无意义的填充数据
        return mask  # (step, step)

    @property
    def attentions(self):
        attentions = {
            "encoder": [l.mh.attention.numpy() for l in self.encoder.ls],
        }
        return attentions


def train(model, data, step=10000, name="gpt"):
    t0 = time.time()
    for t in range(step):
        seqs, segs, xlen, nsp_labels = data.sample(16)
        loss, pred = model.step(seqs[:, :-1], segs[:, :-1], seqs[:, 1:], nsp_labels)
        if t % 100 == 0:
            pred = pred[0].numpy().argmax(axis=1)
            t1 = time.time()
            print(
                "\n\nstep: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss.numpy(),
                "\n| tgt: ", " ".join([data.i2v[i] for i in seqs[0, 1:][:xlen[0].sum() + 1]]),
                "\n| prd: ", " ".join([data.i2v[i] for i in pred[:xlen[0].sum() + 1]]),
            )
            t0 = t1
    os.makedirs("./visual/models/%s" % name, exist_ok=True)
    model.save_weights("./visual/models/%s/model.ckpt" % name)


def export_attention(model, data, name="gpt"):
    model.load_weights("./visual/models/%s/model.ckpt" % name)

    # save attention matrix for visualization
    seqs, segs, xlen, nsp_labels = data.sample(32)
    model.call(seqs[:, :-1], segs[:, :-1], False)
    data = {"src": [[data.i2v[i] for i in seqs[j]] for j in range(len(seqs))], "attentions": model.attentions}
    path = "./visual/tmp/%s_attention_matrix.pkl" % name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    utils.set_soft_gpu(True)
    MODEL_DIM = 256
    N_LAYER = 4
    LEARNING_RATE = 1e-4

    d = utils.MRPCData("./MRPC", 2000)
    print("num word: ", d.num_word)
    m = GPT(
        model_dim=MODEL_DIM, max_len=d.max_len - 1, n_layer=N_LAYER, n_head=4, n_vocab=d.num_word,
        lr=LEARNING_RATE, max_seg=d.num_seg, drop_rate=0.2, padding_idx=d.pad_id)
    train(m, d, step=5000, name="gpt")
    export_attention(m, d, name="gpt")
