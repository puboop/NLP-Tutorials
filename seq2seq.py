# [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils  # this refers to utils.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)
import tensorflow_addons as tfa


class Seq2Seq(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units, max_pred_len, start_token, end_token):
        """
        enc_v_dim: 输入编码长度（词汇量大小）
        dec_v_dim: 输入解码长度（词汇量大小）
        emb_dim: 嵌入层的维度（每个词的嵌入向量的维度）
        units: LSTM单元的数量（隐层大小）
        max_pred_len: 最大预测长度（预测序列的最大长度）
        start_token: 解码器输入的起始标记
        end_token: 解码器输出的终止标记
        """
        super().__init__()
        self.units = units

        # encoder
        """
        enc_embeddings: 定义了一个嵌入层，将输入的词索引转换为嵌入向量
        input_dim 是输入词汇表的大小，output_dim 是嵌入向量的维度
        embeddings_initializer 用于初始化嵌入向量
        """
        self.enc_embeddings = keras.layers.Embedding(
            input_dim=enc_v_dim, output_dim=emb_dim,  # [enc_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        # 定义了一个LSTM层，用于处理嵌入向量序列并生成编码输出和状态
        # units指定了LSTM的单元数量
        self.encoder = keras.layers.LSTM(units=units, return_sequences=True, return_state=True)

        # decoder
        # dec_embeddings 类似于编码器的嵌入层，用于解码器。它将解码器的输入词索引转换为嵌入向量。
        self.dec_embeddings = keras.layers.Embedding(
            input_dim=dec_v_dim, output_dim=emb_dim,  # [dec_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        # decoder_cell定义了LSTM解码器单元，用于生成解码器的输出。
        self.decoder_cell = keras.layers.LSTMCell(units=units)
        # decoder_dense: 一个全连接层，用于将LSTM的输出转换为目标词汇表的大小。
        decoder_dense = keras.layers.Dense(dec_v_dim)

        # train decoder
        # self.decoder_train: 使用TrainingSampler的解码器，用于训练期间。
        # BasicDecoder是TensorFlow Addons中的一个类，用于构建解码器。
        """
        BasicDecoder 的核心概念
        Decoder Cell: 这是解码器的核心部分，通常是一个 RNN 单元，如 LSTMCell 或 GRUCell。它负责在每个时间步更新隐藏状态并生成输出。
        Sampler: Sampler 决定了在每个时间步输入给 Decoder Cell 的是哪个嵌入。不同的 Sampler 实现了不同的策略，如贪心搜索（GreedyEmbeddingSampler）、训练采样（TrainingSampler）等。
        Output Layer: 通常是一个全连接层，将解码器单元的输出映射到目标词汇表的大小，以生成预测。
        
        BasicDecoder 的主要组件和流程
        BasicDecoder 的主要功能是协调解码器单元、采样器和输出层，在每个时间步执行以下操作：
        输入处理: 从 Sampler 获取当前时间步的输入（通常是上一个时间步的输出或初始输入，如起始标记）。
        状态更新: 使用解码器单元更新隐藏状态。
        输出计算: 通过输出层计算当前时间步的输出，通常是目标词汇表的概率分布。
        输出和状态的存储: 将输出和状态存储起来，以便在下一个时间步使用。
        """
        self.decoder_train = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.TrainingSampler(),  # sampler for train
            output_layer=decoder_dense
        )
        """
        训练采样 (TrainingSampler) 的核心概念
        在序列到序列（Seq2Seq）模型的训练过程中，使用 TrainingSampler 可以将目标序列的实际值作为解码器的输入，而不是依赖解码器的先前输出。
        这种方法可以帮助模型更快地学习，因为它避免了错误的传播（即，如果模型预测错误，该错误可能会被传递到后续的时间步）
        TrainingSampler 的主要功能
        采样策略:
        TrainingSampler 使用目标序列的真实值作为输入，不依赖于解码器的先前预测。这有助于稳定训练过程，特别是在模型还没有足够好的情况下。
        动态解码支持:
        TrainingSampler 支持动态解码，这意味着解码过程可以在运行时根据需要灵活地调整，允许处理可变长度的序列。
        """

        # predict decoder
        # self.decoder_eval: 使用GreedyEmbeddingSampler的解码器，用于预测期间。
        # GreedyEmbeddingSampler用于选择下一个单词时进行贪心搜索。
        self.decoder_eval = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(),       # sampler for predict
            output_layer=decoder_dense
        )
        """
        贪心搜索(GreedyEmbeddingSampler) 是 TensorFlow Addons 中用于序列到序列（Seq2Seq）模型的采样器类之一。它在推理（预测）阶段使用，负责在每个时间步选择概率最高的词作为输入。
        GreedyEmbeddingSampler 的主要功能是在没有目标序列（ground truth）可用的情况下，从模型的输出中进行选择，这对于生成文本等任务非常重要。
        GreedyEmbeddingSampler 的核心概念
        在序列到序列模型的解码过程中，每个时间步都会生成一个词的概率分布。GreedyEmbeddingSampler 在每个时间步选择概率最高的词（即 argmax），然后将该词的嵌入作为下一时间步的输入。
        这种贪心策略有助于生成连贯的输出，但它并不一定总能找到全局最优的解码路径，因为每一步的选择都是局部最优的。
        """

        # self.cross_entropy: 使用稀疏类别交叉熵损失函数来计算损失。
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # self.opt: 使用Adam优化器来更新模型参数。
        self.opt = keras.optimizers.Adam(0.01)
        # self.max_pred_len: 记录最大预测长度。
        self.max_pred_len = max_pred_len
        # self.start_token 和 self.end_token: 分别表示解码器的开始标记和结束标记。
        self.start_token = start_token
        self.end_token = end_token

    def encode(self, x):
        # 将输入的词索引转换为嵌入向量
        # 是输入序列的索引表示（例如，每个词被映射为一个唯一的整数）
        # self.enc_embeddings 是一个嵌入层（keras.layers.Embedding），它将这些整数索引转换为对应的嵌入向量（dense vector）。
        # 嵌入表示捕捉了词汇的语义信息，并将其表示为一个固定大小的向量。
        embedded = self.enc_embeddings(x)
        # 这是一个包含两个张量的列表，每个张量的形状为 (batch_size, units)。这两个张量分别是 LSTM 单元的初始隐藏状态 (h) 和初始细胞状态 (c)。
        # x.shape[0] 代表批次大小（即有多少输入序列），self.units 代表 LSTM 单元的维度。
        # tf.zeros((x.shape[0], self.units)) 用零来初始化这些状态，这通常表示模型在开始时没有任何记忆或上下文。
        init_s = [tf.zeros((x.shape[0], self.units)), tf.zeros((x.shape[0], self.units))]
        # 这行代码将嵌入的输入 embedded 传递给编码器 self.encoder。self.encoder 是一个 LSTM 层（keras.layers.LSTM），它能够处理输入序列并生成相应的输出和状态。
        # initial_state=init_s 将前面定义的初始状态传递给 LSTM 层，告诉它从这些初始状态开始计算。
        # o 是所有时间步的输出序列，它包含了每个时间步的输出（即 LSTM 的隐藏状态）。
        # h 是最后一个时间步的隐藏状态（hidden state）。
        # c 是最后一个时间步的细胞状态（cell state）。
        o, h, c = self.encoder(embedded, initial_state=init_s)
        return [h, c]

    def inference(self, x):
        """
        用于生成预测的序列。它通过使用解码器的评估模式（self.decoder_eval）来逐步生成序列
        """
        # 传入 x 传递给编码器，通过 self.encode(x) 计算编码器的最终状态 s。
        # 这个状态包含了编码器对输入序列的压缩表示，将作为解码器生成序列时的初始状态。
        s = self.encode(x)

        # self.decoder_eval.initialize 方法用于初始化解码器在评估阶段的状态。它的参数包括：
        # self.dec_embeddings.variables[0]: 解码器的嵌入层权重，用于生成词汇的嵌入表示。
        # start_tokens=tf.fill([x.shape[0], ], self.start_token): 初始化解码器的输入序列为 <GO> 标记（或开始标记），x.shape[0] 表示批次大小，self.start_token 是开始标记的词汇索引。
        # end_token=self.end_token: 解码器生成的序列中用来指示结束的标记。
        # initial_state=s: 编码器生成的初始状态，传递给解码器。
        done, i, s = self.decoder_eval.initialize(
            self.dec_embeddings.variables[0],
            start_tokens=tf.fill([x.shape[0], ], self.start_token),
            end_token=self.end_token,
            initial_state=s,
        )
        # self.decoder_eval.initialize 返回三个值：
        # done: 一个指示序列是否完成的标志。
        # i: 初始的输入（通常是开始标记的嵌入表示）。
        # s: 解码器的初始状态。

        # 用于存储生成的预测序列。
        # 数组的形状是 (batch_size, max_pred_len)，其中 batch_size 是输入序列的数量，max_pred_len 是最大预测长度。
        pred_id = np.zeros((x.shape[0], self.max_pred_len), dtype=np.int32)
        for l in range(self.max_pred_len):
            # 这是一个循环，用于生成预测序列。循环迭代次数等于 self.max_pred_len，即最大预测长度。
            # 每次迭代代表解码器的一个时间步：
            # self.decoder_eval.step 方法执行解码器的一个时间步。它的参数包括：
            # time=l: 当前时间步（序列位置）。
            # inputs=i: 当前时间步的输入，开始时是 <GO> 标记的嵌入，后续时间步是上一个时间步生成的词汇。
            # state=s: 当前时间步的解码器状态。
            # training=False: 表示当前是在评估阶段，不进行训练。
            # self.decoder_eval.step 返回四个值：
            # o: 解码器的输出，包括每个时间步的预测分布。
            # s: 更新后的解码器状态。
            # i: 更新后的输入。
            # done: 指示序列是否完成的标志。
            o, s, i, done = self.decoder_eval.step(time=l,
                                                   inputs=i,
                                                   state=s,
                                                   training=False
                                                   )
            # pred_id[:, l] = o.sample_id: 将当前时间步的预测词汇索引存储在 pred_id 数组中。
            # o.sample_id 是解码器生成的词汇的索引。
            pred_id[:, l] = o.sample_id
        return pred_id

    def train_logits(self, x, y, seq_len):
        s = self.encode(x)
        # dec_in 是目标序列 y 的一个子集，忽略了序列中的 <EOS>（结束标记）元素。
        # 这是因为在训练过程中，我们通常不希望解码器将 <EOS> 标记作为输入，而是希望它生成该标记作为输出。
        dec_in = y[:, :-1]  # ignore <EOS>
        # self.dec_embeddings 是一个嵌入层（keras.layers.Embedding），它将目标序列中的词索引转换为嵌入向量。
        # 这些嵌入向量作为解码器的输入，提供了词汇的语义信息。
        dec_emb_in = self.dec_embeddings(dec_in)

        # self.decoder_train 采用以下参数：
        ### dec_emb_in: 解码器输入的嵌入表示。
        ### s: 编码器的最终状态，作为解码器的初始状态。
        ### sequence_length=seq_len: 目标序列的实际长度，用于控制解码器的解码步数。
        # self.decoder_train 是一个 BasicDecoder 实例，它使用训练采样器（TrainingSampler）来指导解码器输出。
        # 返回值 o 包含了解码器的输出信息，包括每个时间步的输出 logits。
        o, _, _ = self.decoder_train(dec_emb_in, s, sequence_length=seq_len)
        # o.rnn_output 是 BasicDecoder 输出对象 o 中的一个属性，它包含了解码器在每个时间步的输出 logits。
        # 这些 logits 是在目标词汇表上的未归一化概率分布，表示解码器在每个时间步对词汇的预测。
        logits = o.rnn_output
        return logits

    def step(self, x, y, seq_len):
        with tf.GradientTape() as tape:
            # self.train_logits(x, y, seq_len) 方法，计算模型在给定输入 x 和目标 y 的情况下的预测输出 logits。
            # seq_len 是目标序列的长度。
            # logits 是模型的输出，表示每个时间步上对目标词汇表的未归一化概率分布。
            logits = self.train_logits(x, y, seq_len)

            # y 中去掉了起始标记 <GO>，得到 dec_out。
            # 在序列到序列模型中，训练过程通常从输入 <GO> 标记开始，并让模型生成其后的序列。
            # 因此，目标序列 y 的第一个元素 <GO> 不用于损失计算，忽略它有助于对齐输入和目标序列。
            dec_out = y[:, 1:]  # ignore <GO>

            # self.cross_entropy 是损失函数，它计算目标序列 dec_out 与模型输出 logits 之间的差异。
            # 交叉熵损失度量了模型预测分布与实际分布之间的差异，是序列到序列任务中常用的损失函数。
            loss = self.cross_entropy(dec_out, logits)

            # tape.gradient() 方法计算损失 loss 相对于所有可训练变量 self.trainable_variables 的梯度。
            # 梯度代表了损失函数对每个变量的敏感度，即在这些变量上的变化如何影响损失值。
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

    model = Seq2Seq(data.num_word,
                    data.num_word,
                    emb_dim=16,
                    units=32,
                    max_pred_len=11,
                    start_token=data.start_token,
                    end_token=data.end_token)

    # training
    for t in range(1500):
        bx, by, decoder_len = data.sample(32)
        loss = model.step(bx, by, decoder_len)
        if t % 70 == 0:
            # by 是一个包含目标序列的张量，其中 by[0] 代表批次中的第一个目标序列。
            # by[0, 1:-1] 表示去掉目标序列中的第一个元素（通常是 <GO> 标记）和最后一个元素（通常是 <EOS> 标记）。
            # data.idx2str 是一个方法，用于将索引序列转换为对应的字符串。
            # target 现在包含了去掉起始和结束标记后的目标序列的字符串表示。
            target = data.idx2str(by[0, 1:-1])

            # bx 是一个包含输入序列的张量，其中 bx[0:1] 表示批次中的第一个输入序列。
            # 这里的 bx[0:1] 可能是一个单独的样本，以便与模型进行推理。
            # model.inference 是一个方法，用于生成模型的预测。
            # pred 包含了模型对于输入序列的预测结果（通常是预测的词汇索引序列）。
            pred = model.inference(bx[0:1])

            # pred[0] 代表模型对第一个输入序列的预测结果。
            # data.idx2str 将预测结果（索引序列）转换为字符串形式。
            # res 是模型生成的预测序列的字符串表示。
            res = data.idx2str(pred[0])

            # bx[0] 是批次中的第一个输入序列。
            # data.idx2str 将输入序列（索引序列）转换为字符串形式。src 是原始输入序列的字符串表示。
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
