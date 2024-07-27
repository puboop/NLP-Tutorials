# [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils  # this refers to utils.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)
import tensorflow_addons as tfa
import pickle


class Seq2Seq(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units, attention_layer_size, max_pred_len, start_token,
                 end_token):
        """
        enc_v_dim: 输入编码长度（词汇量大小）
        dec_v_dim: 输入解码长度（词汇量大小）
        emb_dim: 嵌入层的维度（每个词的嵌入向量的维度）
        units: LSTM单元的数量（隐层大小）
        attention_layer_size: 注意力层的大小，即注意力得分的输出维度
        max_pred_len: 最大预测长度（预测序列的最大长度）
        start_token: 解码器输入的起始标记
        end_token: 解码器输出的终止标记
        """
        super().__init__()
        self.units = units

        # encoder
        self.enc_embeddings = keras.layers.Embedding(
            input_dim=enc_v_dim, output_dim=emb_dim,  # [enc_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        self.encoder = keras.layers.LSTM(units=units, return_sequences=True, return_state=True)

        # decoder
        # 这段代码添加了注意力机制到序列到序列模型中，用于更好地处理序列数据的特征建模。它具体实现了一个带有 Luong 注意力机制的解码器部分
        # 实现 Luong 注意力机制，用于在解码阶段对编码器的输出进行加权平均，从而更好地关注输入序列的特定部分
        # units: 指定注意力层的大小，即注意力得分的维度。
        # memory: 通常指编码器的输出序列（即记忆），在初始化时可以为 None，稍后会设置。
        # memory_sequence_length: 编码器输出序列的实际长度，可以为 None，用于掩码操作。
        # Luong 注意力机制是一种常用的注意力机制，它通过计算解码器状态与编码器输出之间的相似性来生成注意力权重，从而选择性地关注输入序列的不同部分。
        self.attention = tfa.seq2seq.LuongAttention(units, memory=None, memory_sequence_length=None)

        # 注意力包装器
        # 将注意力机制集成到解码器的 RNN 单元中，使解码器能够在每个时间步选择性地关注输入序列的不同部分。
        # cell: 底层 RNN 单元，这里使用的是 LSTMCell。
        # LSTMCell 是 LSTM 网络的一个基本单元，它在解码器中用于生成每个时间步的输出。
        # attention_mechanism: 使用的注意力机制，这里是前面定义的 LuongAttention。
        # attention_layer_size: 注意力层的大小，即注意力得分的输出维度。
        # alignment_history: 是否记录注意力对齐历史，这对可视化注意力权重或进行其他分析非常有用。设置为 True，表示会记录注意力权重。
        self.decoder_cell = tfa.seq2seq.AttentionWrapper(
            cell=keras.layers.LSTMCell(units=units),
            attention_mechanism=self.attention,
            attention_layer_size=attention_layer_size,
            alignment_history=True,  # for attention visualization
        )
        # AttentionWrapper 将注意力机制和 RNN 单元结合在一起，在每个时间步上根据注意力权重对输入进行加权，然后再输入到 RNN 单元中。
        # 这种机制可以让解码器在生成每个输出时，根据输入序列的不同部分进行调整，从而提高模型的灵活性和准确性。

        self.dec_embeddings = keras.layers.Embedding(
            input_dim=dec_v_dim, output_dim=emb_dim,  # [dec_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        decoder_dense = keras.layers.Dense(dec_v_dim)  # output layer

        # train decoder
        self.decoder_train = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.TrainingSampler(),  # sampler for train
            output_layer=decoder_dense
        )
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.05, clipnorm=5.0)

        # predict decoder
        self.decoder_eval = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(),  # sampler for predict
            output_layer=decoder_dense
        )

        # prediction restriction
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token

    def encode(self, x):
        o = self.enc_embeddings(x)
        init_s = [tf.zeros((x.shape[0], self.units)), tf.zeros((x.shape[0], self.units))]
        o, h, c = self.encoder(o, initial_state=init_s)
        return o, h, c

    def set_attention(self, x):
        """
        实现了一个方法 set_attention，用于在序列到序列模型中设置注意力机制。
        它首先对输入 x 进行编码，然后使用注意力机制来调整解码器的初始状态。
        """
        o, h, c = self.encode(x)
        # encoder output for attention to focus
        # 设置注意力机制的记忆，用于在解码时提供参考信息。
        # wrap state by attention wrapper
        # self.attention: 之前定义的 LuongAttention 实例，它包含了注意力机制的实现。
        # setup_memory(o): 将编码器的输出 o 作为注意力机制的记忆。
        # 注意力机制会利用这些信息来计算注意力权重，从而决定解码器在每个时间步上关注输入序列的哪些部分。
        self.attention.setup_memory(o)

        # 为解码器设置初始状态，并将其与注意力机制结合。
        # self.decoder_cell: 之前定义的 AttentionWrapper 实例，它包含了解码器的 RNN 单元和注意力机制。
        # get_initial_state: 获取解码器的初始状态，通常由零向量组成。
        # clone(cell_state=[h, c]): 使用编码器的最终状态 [h, c] 替换初始状态中的 cell_state，从而为解码器提供一个合理的初始状态。
        # 这一步是将编码器的信息传递给解码器的关键步骤。
        s = self.decoder_cell \
            .get_initial_state(batch_size=x.shape[0], dtype=tf.float32) \
            .clone(cell_state=[h, c])
        return s

    def inference(self, x, return_align=False):
        """
        用于在序列到序列模型中进行推理（即在不使用训练数据进行调整的情况下生成输出序列）。
        推理过程使用了注意力机制和贪婪解码策略。
        """
        s = self.set_attention(x)
        """
        初始化解码器，以便开始生成输出序列。
        self.decoder_eval: 使用的是评估模式下的解码器（即不需要训练）。
        self.dec_embeddings.variables[0]: 提供解码器嵌入矩阵的变量（词嵌入矩阵）。
        start_tokens: 指定解码器开始生成序列时的起始标记，通常是一个特定的开始符号 <GO>。
        end_token: 指定解码器在何时停止生成输出的结束标记 <EOS>。
        initial_state=s: 提供解码器的初始状态 s，这是通过之前的注意力机制设置的。
        """
        done, i, s = self.decoder_eval \
            .initialize(self.dec_embeddings.variables[0],
                        start_tokens=tf.fill([x.shape[0], ], self.start_token),
                        end_token=self.end_token,
                        initial_state=s,
                        )
        """
        pred_id: 用于存储每个输入序列的预测输出，初始化为全零矩阵，大小为 (batch_size, max_pred_len)。
        使用一个循环遍历最大预测长度 max_pred_len，在每个时间步 l 上执行解码器的 step 方法：
        o: 输出对象，包含了预测的 sample_id。
        s: 更新的解码器状态。
        i: 输入给下一个时间步的词嵌入向量。
        done: 指示生成是否完成。
        o.sample_id: 当前时间步的预测输出，存储到 pred_id 中。
        """
        pred_id = np.zeros((x.shape[0], self.max_pred_len), dtype=np.int32)
        for l in range(self.max_pred_len):
            o, s, i, done = self.decoder_eval \
                .step(time=l, inputs=i, state=s, training=False)
            pred_id[:, l] = o.sample_id
        """
        根据 return_align 的值返回不同的结果。
        如果 return_align 为 True，返回对齐历史（即注意力权重），这是通过转置 s.alignment_history.stack().numpy() 得到的。
        转置操作使其形状变为 (batch_size, max_pred_len, source_sequence_length)，表示每个输入序列在每个时间步上的注意力分布。
        如果 return_align 为 False，则只返回预测的序列 pred_id。
        在这种情况下，调用 s.alignment_history.mark_used() 来避免 TensorFlow 在未来使用该历史记录时发出警告。
        """
        if return_align:
            return np.transpose(s.alignment_history.stack().numpy(), (1, 0, 2))
        else:
            s.alignment_history.mark_used()  # otherwise gives warning
            return pred_id

    def train_logits(self, x, y, seq_len):
        s = self.set_attention(x)
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
    data = utils.DateData(2000)
    print("Chinese time order: yy/mm/dd ", data.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", data.date_en[:3])
    print("vocabularies: ", data.vocab)
    print("x index sample: \n{}\n{}".format(data.idx2str(data.x[0]), data.x[0]),
          "\ny index sample: \n{}\n{}".format(data.idx2str(data.y[0]), data.y[0]))

    model = Seq2Seq(
        data.num_word, data.num_word, emb_dim=12, units=14, attention_layer_size=16,
        max_pred_len=11, start_token=data.start_token, end_token=data.end_token)

    # training
    for t in range(1000):
        bx, by, decoder_len = data.sample(64)
        loss = model.step(bx, by, decoder_len)
        if t % 70 == 0:
            target = data.idx2str(by[0, 1:-1])
            pred = model.inference(bx[0:1])
            res = data.idx2str(pred[0])
            src = data.idx2str(bx[0])
            print(
                "t: ", t,
                "| loss: %.5f" % loss,
                "| input: ", src,
                "| target: ", target,
                "| inference: ", res,
            )

    pkl_data = {"i2v"  : data.i2v, "x": data.x[:6], "y": data.y[:6],
                "align": model.inference(data.x[:6], return_align=True)}

    with open("./visual/tmp/attention_align.pkl", "wb") as f:
        pickle.dump(pkl_data, f)


if __name__ == "__main__":
    train()
