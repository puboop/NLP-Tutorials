# a modification from [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils  # this refers to utils.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)
import tensorflow_addons as tfa


class CNNTranslation(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units, max_pred_len, start_token, end_token):
        super().__init__()
        self.units = units

        # encoder
        # 编码器，相对于seq2seq有着不同通之处，这里是基于cnn卷积网络来做的，所以这里会有卷积核，池化层，全链接层
        self.enc_embeddings = keras.layers.Embedding(
            input_dim=enc_v_dim, output_dim=emb_dim,  # [enc_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        """多个卷积层
        keras.layers.Conv2D：这是一个二维卷积层，适用于处理二维输入数据，如图像或文本的嵌入矩阵。
        16：卷积层的输出通道数（或滤波器数量），意味着每个卷积层会生成 16 个特征图。
        (n, emb_dim)：卷积核的大小，其中 n 是不同的高度，emb_dim 是输入嵌入的维度。这里，n 取值为 2, 3, 4，表示使用不同的卷积核大小来提取不同的 n-gram 特征。
        padding="valid"：表示没有填充，卷积操作会导致输出的空间维度减小。
        activation=keras.activations.relu：ReLU 激活函数，用于引入非线性，有助于模型学习复杂的模式。
        """
        self.conv2ds = [
            keras.layers.Conv2D(16, (n, emb_dim), padding="valid", activation=keras.activations.relu)
            for n in range(2, 5)]
        """多个池化层
        keras.layers.MaxPool2D：二维最大池化层，用于下采样，保留每个池化窗口内的最大值，从而减少特征图的空间尺寸。
        (n, 1)：池化窗口的大小，其中 n 表示高度。池化只在高度方向上进行（n=7, 6, 5），而宽度方向上窗口大小为 1，表示不缩小宽度。
        """
        self.max_pools = [keras.layers.MaxPool2D((n, 1)) for n in [7, 6, 5]]
        """
        keras.layers.Dense：全连接层，用于将输入转换为特定维度的输出向量。
        units：输出的神经元数量，定义了特征表示的维度。
        activation=keras.activations.relu：ReLU 激活函数，进一步引入非线性，并帮助模型学习复杂的模式。
        """
        self.encoder = keras.layers.Dense(units, activation=keras.activations.relu)

        # decoder
        self.dec_embeddings = keras.layers.Embedding(
            input_dim=dec_v_dim, output_dim=emb_dim,  # [dec_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        self.decoder_cell = keras.layers.LSTMCell(units=units)
        decoder_dense = keras.layers.Dense(dec_v_dim)
        # train decoder
        self.decoder_train = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.TrainingSampler(),  # sampler for train
            output_layer=decoder_dense
        )
        # predict decoder
        self.decoder_eval = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(),  # sampler for predict
            output_layer=decoder_dense
        )

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.01)
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token

    def encode(self, x):
        embedded = self.enc_embeddings(x)  # [n, step, emb]
        # 增加一个维度
        o = tf.expand_dims(embedded, axis=3)  # [n, step=8, emb=16, 1]
        # self.conv2ds 包含多个卷积层，每个卷积层使用不同的核大小（高度为2到4，宽度为emb）。
        # 对于每个卷积层，conv2d(o) 计算卷积操作，输出形状为 [n, new_step, 1, 16]，其中 new_step 是卷积后序列的长度，16 是卷积核的数量。
        co = [conv2d(o) for conv2d in self.conv2ds]  # [n, 7, 1, 16], [n, 6, 1, 16], [n, 5, 1, 16]
        # self.max_pools 包含多个最大池化层，对应不同的卷积层。
        # 池化后，特征图的形状变为 [n, 1, 1, 16]，即高度和宽度都被池化到1。
        co = [self.max_pools[i](co[i]) for i in range(len(co))]  # [n, 1, 1, 16] * 3
        # 移除多余的维度，使特征图成为一个向量。
        # tf.squeeze 移除指定的维度，这里移除了高度和宽度维度，使每个特征图变为 [n, 16]。
        co = [tf.squeeze(c, axis=[1, 2]) for c in co]  # [n, 16] * 3
        # 将所有卷积层的特征拼接在一起。
        # tf.concat 在指定的轴上拼接张量，这里在第一个轴（特征维度）拼接，将形状变为 [n, 48]。
        o = tf.concat(co, axis=1)  # [n, 16*3]
        # 将拼接后的特征向量转换为固定大小的表示。
        # self.encoder 是一个全连接层，将输入转换为输出维度为 units 的表示。这里 h 的形状为 [n, units]。
        h = self.encoder(o)  # [n, units]
        return [h, h]

    def inference(self, x):
        s = self.encode(x)
        done, i, s = self.decoder_eval.initialize(self.dec_embeddings.variables[0],
                                                  start_tokens=tf.fill([x.shape[0], ], self.start_token),
                                                  end_token=self.end_token,
                                                  initial_state=s,
                                                  )
        pred_id = np.zeros((x.shape[0], self.max_pred_len), dtype=np.int32)
        for l in range(self.max_pred_len):
            o, s, i, done = self.decoder_eval.step(time=l,
                                                   inputs=i,
                                                   state=s,
                                                   training=False
                                                   )
            pred_id[:, l] = o.sample_id
        return pred_id

    def train_logits(self, x, y, seq_len):
        s = self.encode(x)
        dec_in = y[:, :-1]  # ignore <EOS>
        dec_emb_in = self.dec_embeddings(dec_in)
        o, _, _ = self.decoder_train(dec_emb_in, s, sequence_length=seq_len)
        logits = o.rnn_output
        return logits

    def step(self, x, y, seq_len):
        with tf.GradientTape() as tape:
            logits = self.train_logits(x, y, seq_len)
            dec_out = y[:, 1:]  # ignore <GO>
            loss = self.cross_entropy(dec_out, logits)
            grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()


def train():
    # get and process data
    data = utils.DateData(4000)
    print("Chinese time order: yy/mm/dd ", data.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", data.date_en[:3])
    print("vocabularies: ", data.vocab)
    print("x index sample: \n{}\n{}".format(data.idx2str(data.x[0]), data.x[0]),
          "\ny index sample: \n{}\n{}".format(data.idx2str(data.y[0]), data.y[0]))

    model = CNNTranslation(
        data.num_word, data.num_word, emb_dim=16, units=32,
        max_pred_len=11, start_token=data.start_token, end_token=data.end_token)

    # training
    for t in range(1500):
        bx, by, decoder_len = data.sample(32)
        loss = model.step(bx, by, decoder_len)
        if t % 70 == 0:
            target = data.idx2str(by[0, 1:-1])
            pred = model.inference(bx[0:1])
            res = data.idx2str(pred[0])
            src = data.idx2str(bx[0])
            print(
                "t: ", t,
                "| loss: %.3f" % loss,
                "| input: ", src,
                "| target: ", target,
                "| inference: ", res,
            )


if __name__ == "__main__":
    train()
