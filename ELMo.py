# [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
from tensorflow import keras
import tensorflow as tf
import utils  # this refers to utils.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)
import time
import os


class ELMo(keras.Model):
    def __init__(self, v_dim, emb_dim, units, n_layers, lr):
        """
        v_dim: 词汇表大小
        emb_dim: 嵌入向量的维度
        units: LSTM 层的单元数
        n_layers: LSTM 层的数量
        lr: 学习率
        """
        super().__init__()
        self.n_layers = n_layers
        self.units = units

        # encoder
        # 定义一个嵌入层，将词汇表中的词索引转换为嵌入向量
        # input_dim: 词汇表大小 (v_dim)
        # output_dim: 嵌入向量的维度 (emb_dim)
        # embeddings_initializer: 嵌入向量的初始化器，使用正态分布初始化
        # mask_zero: 指示是否应跳过填充值为零的时间步
        self.word_embed = keras.layers.Embedding(
            input_dim=v_dim, output_dim=emb_dim,  # [n_vocab, emb_dim]
            embeddings_initializer=keras.initializers.RandomNormal(0., 0.001),
            mask_zero=True,
        )

        # forward lstm
        # 定义一组前向 LSTM 层和输出层。
        # 前向 LSTM 层: 一个列表 self.fs 包含 n_layers 个 LSTM 层，每个层都有 units 个单元，return_sequences=True 表示返回每个时间步的输出。
        # 输出层: 一个全连接层 self.f_logits，将 LSTM 的输出转换为词汇表大小的概率分布
        self.fs = [keras.layers.LSTM(units, return_sequences=True) for _ in range(n_layers)]
        self.f_logits = keras.layers.Dense(v_dim)

        # backward lstm
        # 定义一组后向 LSTM 层和输出层。
        # 后向 LSTM 层: 一个列表 self.bs 包含 n_layers 个 LSTM 层，每个层都有 units 个单元，go_backwards=True 表示反向处理序列。
        # 输出层: 一个全连接层 self.b_logits，将 LSTM 的输出转换为词汇表大小的概率分布。
        self.bs = [keras.layers.LSTM(units, return_sequences=True, go_backwards=True) for _ in range(n_layers)]
        self.b_logits = keras.layers.Dense(v_dim)

        # self.cross_entropy1: 前向 LSTM 的稀疏分类交叉熵损失。
        # self.cross_entropy2: 后向 LSTM 的稀疏分类交叉熵损失。
        # from_logits=True: 表示输入是未经过 softmax 激活的 logits。
        # 优化器: 使用 Adam 优化器 self.opt，学习率为 lr。
        self.cross_entropy1 = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.cross_entropy2 = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(lr)

    def call(self, seqs):
        # call方法接受一个输入序列seqs，并通过self.word_embed（嵌入层）将这些序列转换为嵌入向量，形状为[batch_size, sequence_length, embedding_dim]。
        embedded = self.word_embed(seqs)  # [n, step, dim]
        """这段注释解释了前向和后向LSTM的处理流程。它表示前向LSTM从0到3处理序列，并预测1到4；后向LSTM从1到4处理序列，并预测0到3。
        0123    forward
        1234    forward predict
         1234   backward 
         0123   backward predict
        """
        # 计算嵌入层的掩码（mask），用于忽略填充的位置。
        mask = self.word_embed.compute_mask(seqs)
        # fxs初始化为嵌入向量去掉最后一个时间步（用于前向LSTM）。
        # bxs初始化为嵌入向量去掉第一个时间步（用于后向LSTM）。
        fxs, bxs = [embedded[:, :-1]], [embedded[:, 1:]]
        for fl, bl in zip(self.fs, self.bs):
            # 循环遍历前向和后向的LSTM层。
            # 对于每一层：
            # 使用前一层的输出fxs[-1]作为当前层的输入，并计算前向LSTM的输出fx。使用掩码mask[:, :-1]，初始状态由fl.get_initial_state(fxs[-1])获得。
            # 使用前一层的输出bxs[-1]作为当前层的输入，并计算后向LSTM的输出bx。使用掩码mask[:, 1:]，初始状态由bl.get_initial_state(bxs[-1])获得。
            # 将前向LSTM的输出fx添加到fxs列表中。
            # 将后向LSTM的输出bx反转（按时间步轴）后添加到bxs列表中。
            fx = fl(fxs[-1], mask=mask[:, :-1], initial_state=fl.get_initial_state(fxs[-1]))  # [n, step-1, dim]
            bx = bl(bxs[-1], mask=mask[:, 1:], initial_state=bl.get_initial_state(bxs[-1]))  # [n, step-1, dim]
            fxs.append(fx)  # predict 1234
            bxs.append(tf.reverse(bx, axis=[1]))  # predict 0123
        return fxs, bxs

    def step(self, seqs):
        with tf.GradientTape() as tape:
            # 调用call方法进行前向传播，得到前向和后向LSTM的所有层的输出fxs和bxs。
            fxs, bxs = self.call(seqs)
            # 前向和后向LSTM的最后一层输出分别通过全连接层self.f_logits和self.b_logits，得到前向输出fo和后向输出bo。
            fo, bo = self.f_logits(fxs[-1]), self.b_logits(bxs[-1])
            # self.cross_entropy1(seqs[:, 1:], fo)：使用前向输出fo和输入序列的后半部分（从第二个时间步开始）计算前向损失。
            # self.cross_entropy2(seqs[:, :-1], bo)：使用后向输出bo和输入序列的前半部分（去掉最后一个时间步）计算后向损失。
            loss = (self.cross_entropy1(seqs[:, 1:], fo) + self.cross_entropy2(seqs[:, :-1], bo)) / 2
        # tf.GradientTape计算损失相对于所有可训练变量的梯度grads。
        grads = tape.gradient(loss, self.trainable_variables)
        # 使用优化器self.opt应用这些梯度更新模型参数。
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, (fo, bo)

    def get_emb(self, seqs):
        """
        实现了一个获取嵌入的方法。
        它通过前向和后向LSTM层的嵌入，拼接前向和后向LSTM的输出，并将其转换为NumPy数组。最终返回这些拼接的嵌入，用于进一步的分析和处理。
        """
        fxs, bxs = self.call(seqs)
        # from word embedding
        # xs列表存储了前向和后向LSTM层的拼接嵌入。
        # 对于词嵌入层（即第一个LSTM层）：
        # 使用tf.concat沿着最后一个维度（即特征维度）拼接前向LSTM输出fxs[0]的后半部分（去掉第一个时间步）和后向LSTM输出bxs[0]的前半部分（去掉最后一个时间步）。
        # 使用.numpy()方法将拼接的Tensor转换为NumPy数组，并将其添加到xs列表中。
        xs = [tf.concat((fxs[0][:, 1:, :], bxs[0][:, :-1, :]), axis=2).numpy()] + \
             [tf.concat((f[:, :-1, :], b[:, 1:, :]), axis=2).numpy()
              for f, b in zip(fxs[1:], bxs[1:])]  # from sentence embedding
        for x in xs:
            print("layers shape=", x.shape)
        return xs


def train(model, data, step):
    t0 = time.time()
    for t in range(step):
        seqs = data.sample(BATCH_SIZE)
        loss, (fo, bo) = model.step(seqs)
        if t % 80 == 0:
            fp = fo[0].numpy().argmax(axis=1)
            bp = bo[0].numpy().argmax(axis=1)
            t1 = time.time()
            print(
                "\n\nstep: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss.numpy(),
                "\n| tgt: ", " ".join([data.i2v[i] for i in seqs[0] if i != data.pad_id]),
                "\n| f_prd: ", " ".join([data.i2v[i] for i in fp if i != data.pad_id]),
                "\n| b_prd: ", " ".join([data.i2v[i] for i in bp if i != data.pad_id]),
            )
            t0 = t1
    os.makedirs("./visual/models/elmo", exist_ok=True)
    model.save_weights("./visual/models/elmo/model.ckpt")


def export_w2v(model, data):
    model.load_weights("./visual/models/elmo/model.ckpt")
    emb = model.get_emb(data.sample(4))
    print(emb)


if __name__ == "__main__":
    utils.set_soft_gpu(True)
    UNITS = 256
    N_LAYERS = 2
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-3
    d = utils.MRPCSingle("./MRPC", rows=2000)
    print("num word: ", d.num_word)
    m = ELMo(d.num_word, emb_dim=UNITS, units=UNITS, n_layers=N_LAYERS, lr=LEARNING_RATE)
    train(m, d, 10000)
    export_w2v(m, d)
