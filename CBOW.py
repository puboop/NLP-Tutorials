# [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
from tensorflow import keras
import tensorflow as tf
from utils import process_w2v_data  # this refers to utils.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)
from visual import show_w2v_word_embedding  # this refers to visual.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)

corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]


class CBOW(keras.Model):
    def __init__(self, v_dim, emb_dim):
        super().__init__()
        # 词汇表的大小，即词汇量（vocabulary size）
        self.v_dim = v_dim
        # 定义一个嵌入层，用于将离散的词汇索引映射为连续的向量表示
        self.embeddings = keras.layers.Embedding(
            input_dim=v_dim, output_dim=emb_dim,  # [n_vocab, emb_dim]
            # 初始化嵌入向量的权重，使用均值为0、标准差为0.1的正态分布。
            embeddings_initializer=keras.initializers.RandomNormal(0., 0.1),
        )

        # noise-contrastive estimation
        # 定义NCE的权重矩阵，用于负采样
        self.nce_w = self.add_weight(
            name="nce_w", shape=[v_dim, emb_dim],# 权重矩阵的形状为 [词汇表大小, 嵌入向量维度]。
            initializer=keras.initializers.TruncatedNormal(0., 0.1))  # [n_vocab, emb_dim]
        # 定义NCE的偏置向量，用于负采样。
        self.nce_b = self.add_weight(
            name="nce_b", shape=(v_dim,),# 偏置向量的形状为 [词汇表大小]。
            initializer=keras.initializers.Constant(0.1))  # [n_vocab, ]
        # Adam优化器，学习率为0.01。
        self.opt = keras.optimizers.Adam(0.01)

    def call(self, x, training=None, mask=None):
        """
        x：输入数据，形状为 [n, skip_window*2]，其中 n 是批次大小，skip_window*2 是上下文窗口的大小。
        training 和 mask 是可选参数，通常用于区分训练和推理模式，以及掩码操作。
        """
        # x.shape = [n, skip_window*2]
        # 输入 x 通过嵌入层，得到形状为 [n, skip_window*2, emb_dim] 的嵌入向量。
        o = self.embeddings(x)          # [n, skip_window*2, emb_dim]
        # 对嵌入向量在轴1上取平均，得到形状为 [n, emb_dim] 的平均嵌入向量。
        o = tf.reduce_mean(o, axis=1)   # [n, emb_dim]
        return o

    # negative sampling: take one positive label and num_sampled negative labels to compute the loss
    # in order to reduce the computation of full softmax
    def loss(self, x, y, training=None):
        """
        x：输入数据。
        y：目标标签。
        training：可选参数，指示是否在训练模式下
        """
        # 计算输入 x 的嵌入向量
        embedded = self.call(x, training)
        """
        tf.expand_dims(y, axis=1)：将目标标签 y 扩展一个维度，以匹配 nce_loss 的输入要求。
        tf.nn.nce_loss：计算负采样损失。
            weights=self.nce_w：NCE的权重矩阵。
            biases=self.nce_b：NCE的偏置向量。
            labels=tf.expand_dims(y, axis=1)：目标标签。
            inputs=embedded：输入嵌入向量。
            num_sampled=5：负采样数量，表示每个正样本配5个负样本。
            num_classes=self.v_dim：词汇表大小
        返回平均的负采样损失。
        """
        return tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.nce_w, biases=self.nce_b, labels=tf.expand_dims(y, axis=1),
                inputs=embedded, num_sampled=5, num_classes=self.v_dim))

    def step(self, x, y):
        # TensorFlow 2.x中的自动微分机制（tf.GradientTape）来计算损失函数的梯度
        # 记录操作以便自动求导的上下文管理器
        with tf.GradientTape() as tape:
            # x 和 y 是输入数据和目标标签
            loss = self.loss(x, y, True)
            # 使用 tape.gradient 计算相对于 self.trainable_variables（模型的可训练参数）的损失梯度。
            # self.trainable_variables是一个包含模型所有可训练变量的列表
            grads = tape.gradient(loss, self.trainable_variables)
        # zip(grads, self.trainable_variables) 将梯度和变量打包在一起，使优化器知道如何更新每个变量。
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()


def train(model, data):
    for t in range(2500):
        bx, by = data.sample(8)
        loss = model.step(bx, by)
        if t % 200 == 0:
            print("step: {} | loss: {}".format(t, loss))


if __name__ == "__main__":
    d = process_w2v_data(corpus, skip_window=2, method="cbow")
    m = CBOW(d.num_word, 2)
    train(m, d)

    # plotting
    show_w2v_word_embedding(m, d, "./visual/results/cbow.png")