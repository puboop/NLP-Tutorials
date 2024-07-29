# [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils  # this refers to utils.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)
import time
import pickle
import os

MODEL_DIM = 32
MAX_LEN = 12
N_LAYER = 3
N_HEAD = 4
DROP_RATE = 0.1


class MultiHead(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        """
        实现了一个多头注意力机制（MultiHead Attention）的类 MultiHead，它是 Transformer 模型中的一个关键组件。多头注意力机制能够在不同的注意力头中并行计算注意力分数，从而捕获输入序列中的不同特征。

        n_head: 注意力头的数量。每个头独立计算不同的注意力分数。
        model_dim: 输入的特征维度。
        drop_rate: Dropout 率，用于防止过拟合
        """
        super().__init__()
        #  每个注意力头的维度，计算为 model_dim / n_head。
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim

        # self.wq, self.wk, self.wv: 分别是用于计算查询（query）、键（key）和值（value）的线性变换层。每个变换的输出维度为 n_head * head_dim。
        self.wq = keras.layers.Dense(n_head * self.head_dim)
        self.wk = keras.layers.Dense(n_head * self.head_dim)
        self.wv = keras.layers.Dense(n_head * self.head_dim)  # [n, step, h*h_dim]

        # self.o_dense: 用于将多头注意力的输出映射回 model_dim 大小的线性层。
        self.o_dense = keras.layers.Dense(model_dim)
        # self.o_drop: Dropout 层，用于防止过拟合。
        self.o_drop = keras.layers.Dropout(rate=drop_rate)
        # self.attention: 用于存储注意力权重。
        self.attention = None

    def call(self, q, k, v, mask, training):
        """
        q, k, v: 分别代表查询、键和值的张量，形状为 [n, step, dim]，其中 n 是批次大小，step 是序列长度，dim 是特征维度。
        mask: 遮掩张量，用于在注意力计算中忽略特定位置。
        training: 布尔标志，指示是否处于训练模式。
        """
        # 线性变换:
        # 对 q、k 和 v 分别应用线性变换得到 _q, _k, _v。形状变为 [n, step, n_head * head_dim]。
        _q = self.wq(q)  # [n, q_step, h*h_dim]
        _k, _v = self.wk(k), self.wv(v)  # [n, step, h*h_dim]

        # 分割为多个头:
        # 使用 split_heads 方法将张量分割为多个注意力头，形状变为 [n, n_head, step, head_dim]。
        _q = self.split_heads(_q)  # [n, h, q_step, h_dim]
        _k, _v = self.split_heads(_k), self.split_heads(_v)  # [n, h, step, h_dim]

        # 计算注意力:
        # 使用 scaled_dot_product_attention 方法计算注意力分数和上下文向量 context。
        context = self.scaled_dot_product_attention(_q, _k, _v, mask)  # [n, q_step, h*dv]

        # 线性变换和Dropout:
        # 将上下文向量 context 通过线性层 self.o_dense 映射回 model_dim 大小，并应用 Dropout。
        o = self.o_dense(context)  # [n, step, dim]
        o = self.o_drop(o, training=training)
        return o

    def split_heads(self, x):
        # split_heads 方法将输入张量 x 分割为多个头。
        # 首先，改变张量的形状以包含头的数量，然后通过 tf.transpose 重新排列轴的顺序，以便将头的数量放在第二个维度。
        # 最终输出的形状为 [n, n_head, step, head_dim]。
        x = tf.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))  # [n, step, h, h_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [n, h, step, h_dim]

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # 计算注意力分数
        # 通过点积计算注意力分数，并进行缩放（除以 sqrt(dk)）以防止梯度消失。dk 是键张量的最后一个维度大小。
        dk = tf.cast(k.shape[-1], dtype=tf.float32)

        # 应用遮掩:
        # 如果提供了 mask，将其应用于注意力分数，确保被遮掩的部分不会影响注意力权重的计算。
        score = tf.matmul(q, k, transpose_b=True) / (tf.math.sqrt(dk) + 1e-8)  # [n, h_dim, q_step, step]
        if mask is not None:
            score += mask * -1e9

        # 计算注意力权重:
        # 通过 softmax 函数对分数进行归一化，得到注意力权重 self.attention。
        self.attention = tf.nn.softmax(score, axis=-1)  # [n, h, q_step, step]
        context = tf.matmul(self.attention, v)  # [n, h, q_step, step] @ [n, h, step, dv] = [n, h, q_step, dv]

        # 计算上下文向量:
        # 将注意力权重与值 v 进行矩阵乘法，得到上下文向量 context，并重新排列轴顺序，最后调整形状以匹配预期输出。
        context = tf.transpose(context, perm=[0, 2, 1, 3])  # [n, q_step, h, dv]
        context = tf.reshape(context, (context.shape[0], context.shape[1], -1))  # [n, q_step, h*dv]
        return context


class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim):
        """
        PositionWiseFFN 代表位置编码感知的前馈神经网络（Feed-Forward Neural Network），通常用于 Transformer 模型中的每一层。这个网络的作用是对输入的每个位置单独进行非线性变换，增加模型的表达能力。

        model_dim: 输入数据的特征维度。
        dff: 前馈神经网络中隐藏层的大小，通常是 model_dim 的 4 倍，以增加模型的复杂度。
        """
        super().__init__()
        dff = model_dim * 4
        # self.l: 全连接层（Dense layer），输出大小为 dff，使用 ReLU 激活函数。这是网络的隐藏层。
        self.l = keras.layers.Dense(dff, activation=keras.activations.relu)
        # self.o: 全连接层，输出大小为 model_dim，不使用激活函数。这是网络的输出层。
        self.o = keras.layers.Dense(model_dim)

    def call(self, x):
        # x: 输入张量，形状为 [n, step, model_dim]，其中 n 是批次大小，step 是序列长度，model_dim 是特征维度。
        """
        工作流程
        隐藏层计算:
        o = self.l(x): 将输入 x 通过第一层全连接层 self.l，并应用 ReLU 激活函数。输出的形状为 [n, step, dff]。
        输出层计算:
        o = self.o(o): 将中间结果 o 通过第二层全连接层 self.o，将输出维度调整回 model_dim，形状为 [n, step, model_dim]。
        返回值:
        返回最终的输出 o，形状与输入相同，即 [n, step, model_dim]。
        """
        o = self.l(x)
        o = self.o(o)
        return o  # [n, step, dim]


class EncodeLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        """
        实现 Transformer 模型中的编码层。编码层是 Transformer 模型的一个基本组成部分，主要由多头自注意力机制、前馈神经网络和归一化层组成。

        n_head: 多头自注意力机制中的头数。多个注意力头允许模型从不同的子空间中捕获不同的信息。
        model_dim: 模型的隐藏层维度，即输入和输出的特征维度。
        drop_rate: Dropout 层的丢弃率，用于防止过拟合。
        """
        super().__init__()
        # 两个 LayerNormalization 层的列表，用于对输入的最后一个轴（即特征维度）进行归一化，帮助稳定和加速训练过程
        self.ln = [keras.layers.LayerNormalization(axis=-1) for _ in range(2)]  # only norm z-dim
        # 自定义的多头自注意力机制（MultiHead）模块，能够让模型关注输入序列的不同部分。
        self.mh = MultiHead(n_head, model_dim, drop_rate)
        # 自定义的前馈神经网络（PositionWiseFFN）模块，在每个位置独立地应用前馈网络，常用于提高模型的表达能力。
        self.ffn = PositionWiseFFN(model_dim)
        #  Dropout 层，用于防止过拟合。
        self.drop = keras.layers.Dropout(drop_rate)

    def call(self, xz, training, mask):
        """
        xz: 输入的张量，通常是输入序列的嵌入表示，形状为 [n, step, dim]，其中 n 是批次大小，step 是序列长度，dim 是特征维度。
        training: 一个布尔标志，指示当前是否处于训练模式。在训练模式下，Dropout 等正则化技术将启用。
        mask: 掩码张量，用于屏蔽输入中的无效部分，通常是填充部分，防止它们对模型的注意力计算产生影响。

        编码层的工作流程
        多头自注意力机制: 通过 self.mh 计算多头自注意力，输入为 xz，注意力权重的计算也基于 xz。输出 attn 的形状为 [n, step, dim]。
        残差连接与归一化: 将注意力输出 attn 与原输入 xz 相加（残差连接），并通过第一个 LayerNormalization 层 self.ln[0] 进行归一化处理，得到 o1。
        前馈神经网络与Dropout: 通过 self.ffn 应用前馈神经网络，将 o1 作为输入。然后，应用 Dropout 正则化，得到 ffn。
        第二次残差连接与归一化: 将前馈神经网络的输出 ffn 与 o1 相加（残差连接），并通过第二个 LayerNormalization 层 self.ln[1] 进行归一化处理，得到最终输出 o。
        输出: 最终返回的张量 o，其形状为 [n, step, dim]，表示经过编码层处理后的序列表示。
        """
        attn = self.mh.call(xz, xz, xz, mask, training)  # [n, step, dim]
        o1 = self.ln[0](attn + xz)
        ffn = self.drop(self.ffn.call(o1), training)
        o = self.ln[1](ffn + o1)  # [n, step, dim]
        return o


class Encoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        """
        用于序列到序列（Seq2Seq）模型中的编码器部分。
        编码器由多个编码层（EncodeLayer）组成，每个编码层通常包括多头自注意力机制和前馈神经网络。

        n_head: 多头自注意力机制中的头数。多个注意力头允许模型从不同的子空间中捕获不同的信息。
        model_dim: 模型的隐藏层维度，定义了每个层的输出维度。
        drop_rate: Dropout 层的丢弃率，用于防止过拟合。
        n_layer: 编码器中编码层的数量。每个编码层包含一组多头自注意力和前馈网络。
        """
        super().__init__()
        # self.ls: EncodeLayer 实例的列表，每个实例代表编码器中的一个层。
        # EncodeLayer 是自定义的编码层类，通常包括多头自注意力和前馈神经网络。
        self.ls = [EncodeLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def call(self, xz, training, mask):
        """
        xz: 输入到编码器的张量，通常是一个嵌入后的输入序列。
        形状为 [n, step, dim]，其中 n 是批次大小，step 是序列长度，dim 是嵌入维度。
        training: 一个布尔标志，指示当前是否处于训练模式。
        该标志通常用于在训练时启用 Dropout 等正则化技术，而在推理时禁用它们。
        mask: 掩码张量，用于屏蔽掉输入中的填充部分，防止它们对模型的学习产生影响。

        编码器的工作流程
        多层编码: 输入的张量 xz 依次通过每一个编码层 EncodeLayer。每个编码层通过其 call 方法进行处理，应用多头自注意力和前馈神经网络，并返回更新后的张量 xz。
        输出: 最终返回处理后的张量 xz，其形状为 [n, step, dim]。这个输出表示输入序列经过多个编码层后的上下文表示。
        """
        for l in self.ls:
            xz = l.call(xz, training, mask)
        return xz  # [n, step, dim]


class DecoderLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        """
        实现 Transformer 模型中的解码层。解码层是 Transformer 模型中的另一基本组成部分，主要用于处理输入序列的上下文信息并生成输出序列。

        n_head: 多头自注意力机制中的头数。允许模型从不同的子空间中关注不同的信息。
        model_dim: 模型的隐藏层维度，即输入和输出的特征维度。
        drop_rate: Dropout 层的丢弃率，用于防止过拟合。
        """
        super().__init__()
        # 三个 LayerNormalization 层的列表，用于对输入的最后一个轴（即特征维度）进行归一化，帮助稳定和加速训练过程
        self.ln = [keras.layers.LayerNormalization(axis=-1) for _ in range(3)]  # only norm z-dim
        # Dropout 层，用于防止过拟合。
        self.drop = keras.layers.Dropout(drop_rate)
        # 两个多头自注意力机制（MultiHead）模块：
        #### 第一个用于解码器的自注意力。
        #### 第二个用于解码器和编码器之间的交互注意力。
        self.mh = [MultiHead(n_head, model_dim, drop_rate) for _ in range(2)]
        # 前馈神经网络（PositionWiseFFN）模块，在每个位置独立地应用前馈网络。
        self.ffn = PositionWiseFFN(model_dim)

    def call(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        """
        yz: 解码器的输入张量，通常是解码器中的前一个位置的输出或开始标记，形状为 [n, step, dim]，其中 n 是批次大小，step 是序列长度，dim 是特征维度。
        xz: 编码器的输出张量，提供了解码过程中所需的上下文信息。
        training: 一个布尔标志，指示当前是否处于训练模式。在训练模式下，Dropout 等正则化技术将启用。
        yz_look_ahead_mask: 解码器自注意力中的遮掩张量，用于防止模型在预测下一个词时看到未来的信息（即“look-ahead”）。
        xz_pad_mask: 编码器输出的填充遮掩张量，用于忽略输入中的填充部分，防止它们对模型的注意力计算产生影响。

        解码层的工作流程
        解码器自注意力: 通过第一个多头自注意力机制 self.mh[0] 计算解码器的自注意力，输入为 yz，注意力权重的计算也基于 yz，应用 yz_look_ahead_mask 进行遮掩。输出 attn 的形状为 [n, step, dim]。
        残差连接与归一化: 将自注意力输出 attn 与输入 yz 相加（残差连接），并通过第一个 LayerNormalization 层 self.ln[0] 进行归一化处理，得到 o1。
        解码器-编码器注意力: 通过第二个多头自注意力机制 self.mh[1] 计算解码器和编码器之间的注意力，输入为 o1，注意力权重的计算基于 xz，应用 xz_pad_mask 进行遮掩。输出 attn 的形状为 [n, step, dim]。
        第二次残差连接与归一化: 将注意力输出 attn 与 o1 相加（残差连接），并通过第二个 LayerNormalization 层 self.ln[1] 进行归一化处理，得到 o2。
        前馈神经网络与Dropout: 通过 self.ffn 应用前馈神经网络，将 o2 作为输入。然后，应用 Dropout 正则化，得到 ffn。
        第三次残差连接与归一化: 将前馈神经网络的输出 ffn 与 o2 相加（残差连接），并通过第三个 LayerNormalization 层 self.ln[2] 进行归一化处理，得到最终输出 o。
        输出: 最终返回的张量 o，其形状为 [n, step, dim]，表示经过解码层处理后的序列表示。
        """
        attn = self.mh[0].call(yz, yz, yz, yz_look_ahead_mask, training)  # decoder self attention
        o1 = self.ln[0](attn + yz)
        attn = self.mh[1].call(o1, xz, xz, xz_pad_mask, training)  # decoder + encoder attention
        o2 = self.ln[1](attn + o1)
        ffn = self.drop(self.ffn.call(o2), training)
        o = self.ln[2](ffn + o2)
        return o


class Decoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        """
        用于序列到序列（Seq2Seq）模型中的解码器部分。解码器由多个解码层（DecoderLayer）组成，
        每个解码层通常包括多头自注意力机制、编码-解码注意力机制和前馈神经网络。

        n_head: 多头自注意力机制中的头数。多个注意力头允许模型从不同的子空间中捕获不同的信息。
        model_dim: 模型的隐藏层维度，定义了每个层的输出维度。
        drop_rate: Dropout 层的丢弃率，用于防止过拟合。
        n_layer: 解码器中解码层的数量。每个解码层包含一组多头自注意力、编码-解码注意力和前馈网络。
        """
        super().__init__()
        # self.ls: DecoderLayer 实例的列表，每个实例代表解码器中的一个层。
        # DecoderLayer 是自定义的解码层类，通常包括多头自注意力、编码-解码注意力和前馈神经网络。
        self.ls = [DecoderLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def call(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        """
        yz: 解码器的输入张量，通常是目标序列的嵌入表示。形状为 [n, step_y, dim]，其中 n 是批次大小，step_y 是目标序列的长度，dim 是嵌入维度。
        xz: 编码器的输出张量，表示源序列的上下文表示。形状为 [n, step_x, dim]，其中 step_x 是源序列的长度。
        training: 一个布尔标志，指示当前是否处于训练模式。该标志通常用于在训练时启用 Dropout 等正则化技术，而在推理时禁用它们。
        yz_look_ahead_mask: 目标序列的前瞻掩码，用于在训练过程中屏蔽未来的词，防止信息泄漏。它确保每个位置只能看到其之前的位置。
        xz_pad_mask: 源序列的填充掩码，用于屏蔽输入中的填充部分，防止它们对模型的注意力计算产生影响。

        解码器的工作流程
        多层解码: 输入的张量 yz 依次通过每一个解码层 DecoderLayer。每个解码层通过其 call 方法进行处理，应用多头自注意力、编码-解码注意力和前馈神经网络，并返回更新后的张量 yz。
        输出: 最终返回处理后的张量 yz，其形状为 [n, step_y, dim]。这个输出表示目标序列在给定源序列上下文的情况下的表示。
        """
        for l in self.ls:
            yz = l.call(yz, xz, training, yz_look_ahead_mask, xz_pad_mask)
        return yz


class PositionEmbedding(keras.layers.Layer):
    def __init__(self, max_len, model_dim, n_vocab):
        """将输入的词索引转换为词向量并添加位置编码
        max_len: 输入序列的最大长度。
        model_dim: 词嵌入的维度。
        n_vocab: 词汇表的大小。
        """
        super().__init__()

        # pos = np.arange(max_len)[:, None]: 创建一个列向量，其中包含从 0 到 max_len-1 的整数，表示每个位置的索引。
        pos = np.arange(max_len)[:, None]
        # pe = pos / np.power(10000, 2. * np.arange(model_dim)[None, :] / model_dim): 计算位置编码矩阵 pe。
        # 这里使用的是 Transformer 论文中的公式。
        #### np.arange(model_dim)[None, :]: 生成一个包含 0 到 model_dim-1 的行向量。
        #### np.power(10000, 2. * np.arange(model_dim)[None, :] / model_dim): 计算位置编码的分母部分。
        #### pos / ...: 计算每个位置和每个维度的编码值。
        pe = pos / np.power(10000, 2. * np.arange(model_dim)[None, :] / model_dim)  # [max_len, dim]
        # pe[:, 0::2] = np.sin(pe[:, 0::2]): 对偶数维度应用 sin 函数。
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        # pe[:, 1::2] = np.cos(pe[:, 1::2]): 对奇数维度应用 cos 函数。
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        # pe = pe[None, :, :]: 增加一个维度，使其适用于批处理
        pe = pe[None, :, :]  # [1, max_len, model_dim]    for batch adding
        # self.pe = tf.constant(pe, dtype=tf.float32): 将 pe 转换为 TensorFlow 常量。
        self.pe = tf.constant(pe, dtype=tf.float32)

        # 嵌入词向量
        # self.embeddings: 定义一个词嵌入层，将词索引转换为词向量。
        # input_dim=n_vocab: 输入的词汇表大小。
        # output_dim=model_dim: 词嵌入的维度。
        # embeddings_initializer=tf.initializers.RandomNormal(0., 0.01): 使用均值为 0、标准差为 0.01 的正态分布初始化词嵌入。
        self.embeddings = keras.layers.Embedding(
            input_dim=n_vocab, output_dim=model_dim,  # [n_vocab, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        )

    def call(self, x):
        """
        x: 输入的词索引序列，形状为 [batch_size, sequence_length]。
        x_embed = self.embeddings(x) + self.pe:
        self.embeddings(x): 将词索引转换为词嵌入，形状为 [batch_size, sequence_length, model_dim]。
        self.pe: 位置编码，形状为 [1, max_len, model_dim]。
        self.embeddings(x) + self.pe: 将词嵌入和位置编码相加，得到最终的嵌入，形状为 [batch_size, sequence_length, model_dim]。
        return x_embed: 返回加上位置编码后的词嵌入。
        """
        x_embed = self.embeddings(x) + self.pe  # [n, step, dim]
        return x_embed


class Transformer(keras.Model):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, drop_rate=0.1, padding_idx=0):
        """
        model_dim: 模型的隐藏层维度，通常用于定义嵌入层和模型中其他层的输出维度。
        max_len: 输入序列的最大长度。
        n_layer: 编码器和解码器的层数。
        n_head: 多头注意力机制中的头数。
        n_vocab: 词汇表的大小，用于定义嵌入层和输出层的维度。
        drop_rate: Dropout 层的丢弃率，用于防止过拟合。
        padding_idx: 用于表示填充的索引值，通常用于忽略序列中的填充部分。
        """
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx

        # PositionEmbedding: 这是一个自定义的嵌入层类，它结合了词嵌入和位置编码。
        # 该层的目的是将输入的词索引转换为词嵌入，并添加位置编码，以便模型能够区分序列中不同位置的词。
        self.embed = PositionEmbedding(max_len, model_dim, n_vocab)
        # Encoder: 这是一个自定义的编码器类，通常包括多头自注意力机制和前馈神经网络。
        # 编码器的主要功能是处理输入序列并生成上下文表示。
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)
        # Decoder: 这是一个自定义的解码器类，类似于编码器，但解码器通常包括额外的机制来处理从编码器传递的上下文信息。
        # 它主要用于生成目标序列。
        self.decoder = Decoder(n_head, model_dim, drop_rate, n_layer)
        # 全连接层，输出的维度为词汇表大小。这一层将解码器的输出转换为词汇表中的每个词的概率分布。
        self.o = keras.layers.Dense(n_vocab)
        # self.cross_entropy: 使用稀疏类别交叉熵作为损失函数，from_logits=True 表示输入未经过 softmax 激活，reduction="none" 表示不聚合损失值，通常用于序列模型中计算每个时间步的损失。
        # self.opt = keras.optimizers.Adam(0.002): 使用 Adam 优化器来训练模型，学习率设置为 0.002。
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.opt = keras.optimizers.Adam(0.002)

    def call(self, x, y, training=None):
        """
        x: 输入序列。
        y: 目标序列（通常是解码器的输入）。
        training: 是否在训练模式下。

        工作流程
        嵌入:
        x_embed, y_embed: 使用位置编码和词嵌入将输入序列 x 和目标序列 y 转换为嵌入向量。
        编码:
        pad_mask: 计算填充掩码。
        encoded_z: 使用编码器处理输入嵌入，生成编码器的输出。
        解码:
        decoded_z: 使用解码器处理目标嵌入和编码器的输出，生成解码器的输出。
        输出:
        o: 使用全连接层将解码器的输出转换为词汇表大小的概率分布。
        """
        x_embed, y_embed = self.embed(x), self.embed(y)
        pad_mask = self._pad_mask(x)
        encoded_z = self.encoder.call(x_embed, training, mask=pad_mask)
        decoded_z = self.decoder.call(
            y_embed, encoded_z, training, yz_look_ahead_mask=self._look_ahead_mask(y), xz_pad_mask=pad_mask)
        o = self.o(decoded_z)
        return o

    def step(self, x, y):
        # logits: 预测结果。
        # pad_mask: 计算实际标签的掩码。
        # loss: 计算稀疏类别交叉熵损失并聚合。
        with tf.GradientTape() as tape:
            logits = self.call(x, y[:, :-1], training=True)
            pad_mask = tf.math.not_equal(y[:, 1:], self.padding_idx)
            loss = tf.reduce_mean(tf.boolean_mask(self.cross_entropy(y[:, 1:], logits), pad_mask))
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, logits

    def _pad_bool(self, seqs):
        # 返回布尔掩码，用于指示填充位置。
        return tf.math.equal(seqs, self.padding_idx)

    def _pad_mask(self, seqs):
        # 生成填充掩码，以便在计算注意力时忽略填充位置。
        mask = tf.cast(self._pad_bool(seqs), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]  # (n, 1, 1, step)

    def _look_ahead_mask(self, seqs):
        # 生成前瞻掩码，用于解码器的自注意力，防止模型看到未来的词。
        mask = 1 - tf.linalg.band_part(tf.ones((self.max_len, self.max_len)), -1, 0)
        mask = tf.where(self._pad_bool(seqs)[:, tf.newaxis, tf.newaxis, :], 1, mask[tf.newaxis, tf.newaxis, :, :])
        return mask  # (step, step)

    def translate(self, src, v2i, i2v):
        # src: 输入序列。
        # v2i: 词汇到索引的映射。
        # i2v: 索引到词汇的映射。
        """
        工作流程
        初始化:
            对输入序列 src 进行填充，并初始化目标序列 tgt 为 <GO> 标记。
        生成预测:
            在每个时间步，使用模型生成预测，选择概率最高的词。
        结束条件:
            直到生成的目标序列达到最大长度或所有词汇都生成完毕。
        返回结果:
            将生成的索引序列转换为词汇并返回。
        """
        # 将源序列 src 填充到最大长度 self.max_len，确保所有输入序列具有相同的长度。utils.pad_zero 是一个工具函数，用于在序列的末尾填充零（通常是填充标记的 ID）。
        src_pad = utils.pad_zero(src, self.max_len)
        # 初始化目标序列 tgt。每个目标序列的开头都被填充了开始标记 (<GO>)，并且目标序列的长度被填充到 self.max_len + 1，以便后续生成的词可以存放在序列的最后位置。
        tgt = utils.pad_zero(np.array([[v2i["<GO>"], ] for _ in range(len(src))]), self.max_len + 1)
        # 初始化目标序列的时间步索引 tgti，用于跟踪当前生成的序列位置。
        tgti = 0
        # 对填充后的源序列 src_pad 进行词嵌入和位置编码，得到 x_embed。这是编码器的输入。
        x_embed = self.embed(src_pad)
        # 使用编码器处理源序列的嵌入表示 x_embed，生成编码器的上下文表示 encoded_z。_pad_mask 用于遮盖填充位置。
        encoded_z = self.encoder.call(x_embed, False, mask=self._pad_mask(src_pad))
        while True:
            # 在每次迭代中，y 是当前目标序列的所有时间步，但不包括当前时间步 tgti，用于作为解码器的输入。
            y = tgt[:, :-1]
            # 对当前目标序列 y 进行词嵌入和位置编码，得到 y_embed。
            y_embed = self.embed(y)
            # 使用解码器处理目标序列的嵌入表示 y_embed 和编码器的上下文表示 encoded_z，生成解码器的输出 decoded_z。_look_ahead_mask 遮盖未来的位置，_pad_mask 遮盖源序列中的填充位置。
            decoded_z = self.decoder.call(y_embed,
                                          encoded_z,
                                          False,
                                          yz_look_ahead_mask=self._look_ahead_mask(y),
                                          xz_pad_mask=self._pad_mask(src_pad)
                                          )
            # 使用全连接层 self.o 将解码器的输出 decoded_z 转换为词汇表大小的概率分布 logits。选择当前时间步 tgti 的预测结果，并将其转换为 NumPy 数组。
            logits = self.o(decoded_z)[:, tgti, :].numpy()
            # 从 logits 中选择概率最高的索引 idx，作为当前时间步 tgti 的预测词。
            idx = np.argmax(logits, axis=1)
            # 更新目标序列 tgt 中当前时间步 tgti 的位置为预测的词索引 idx，并递增时间步索引 tgti。
            tgti += 1
            tgt[:, tgti] = idx
            if tgti >= self.max_len:
                break
        # 将生成的目标序列 tgt 转换为实际的文本输出。
        # i2v 是从索引到词汇的映射，将目标序列中索引转换为实际词汇，并去掉序列中的 <GO> 标记（即 tgt[j, 1:tgti]）。
        # 返回一个列表，其中每个元素是源序列 src 对应的翻译结果。
        return ["".join([i2v[i] for i in tgt[j, 1:tgti]]) for j in range(len(src))]

    @property
    def attentions(self):
        # 提供了编码器和解码器中每层的注意力权重，方便进行可视化和分析。
        attentions = {
            "encoder": [l.mh.attention.numpy() for l in self.encoder.ls],
            "decoder": {
                "mh1": [l.mh[0].attention.numpy() for l in self.decoder.ls],
                "mh2": [l.mh[1].attention.numpy() for l in self.decoder.ls],
            }}
        return attentions


def train(model, data, step):
    # training
    t0 = time.time()
    for t in range(step):
        # bx by是一个二维列表
        bx, by, seq_len = data.sample(64)
        # 生成一个ndarray，并将其中的二维度填充为bx的值
        bx, by = utils.pad_zero(bx, max_len=MAX_LEN), utils.pad_zero(by, max_len=MAX_LEN + 1)
        loss, logits = model.step(bx, by)
        if t % 50 == 0:
            logits = logits[0].numpy()
            t1 = time.time()
            print(
                "step: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.4f" % loss.numpy(),
                "| target: ", "".join([data.i2v[i] for i in by[0, 1:10]]),
                "| inference: ", "".join([data.i2v[i] for i in np.argmax(logits, axis=1)[:10]]),
            )
            t0 = t1

    os.makedirs("./visual/models/transformer", exist_ok=True)
    model.save_weights("./visual/models/transformer/model.ckpt")
    os.makedirs("./visual/tmp", exist_ok=True)
    with open("./visual/tmp/transformer_v2i_i2v.pkl", "wb") as f:
        pickle.dump({"v2i": data.v2i, "i2v": data.i2v}, f)


def export_attention(model, data, name="transformer"):
    """
    功能: 定义一个函数 export_attention，用于导出注意力矩阵及其他相关数据。
    model: 训练好的 Transformer 模型。
    data: 数据对象，包含用于生成注意力矩阵的源数据和目标数据。
    name: 可选的文件名标识，用于保存注意力矩阵文件的名称。
    """
    # 从磁盘加载一个包含词汇到索引映射 (v2i) 和索引到词汇映射 (i2v) 的字典 dic。这个字典用于将词汇转换为索引和从索引恢复词汇。
    with open("./visual/tmp/transformer_v2i_i2v.pkl", "rb") as f:
        dic = pickle.load(f)
    # 加载模型的权重。这是为了确保在进行注意力矩阵的计算时，模型是处于训练或验证状态的最新权重。
    model.load_weights("./visual/models/transformer/model.ckpt")
    # 从数据集中随机抽取32个样本，分别作为源序列 (bx) 和目标序列 (by)。seq_len 是这些序列的长度。
    bx, by, seq_len = data.sample(32)
    # 使用模型生成目标序列，传入源序列 bx 以及从词汇到索引的映射 (dic["v2i"]) 和从索引到词汇的映射 (dic["i2v"])。
    model.translate(bx, dic["v2i"], dic["i2v"])
    # "src": 源序列的词汇表示列表，将 bx 中的索引转换为词汇。
    # "tgt": 目标序列的词汇表示列表，将 by 中的索引转换为词汇。
    # "attentions": 从模型中提取的注意力矩阵数据，用于可视化。
    attn_data = {
        "src"       : [[data.i2v[i] for i in bx[j]] for j in range(len(bx))],
        "tgt"       : [[data.i2v[i] for i in by[j]] for j in range(len(by))],
        "attentions": model.attentions
    }
    path = "./visual/tmp/%s_attention_matrix.pkl" % name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(attn_data, f)


if __name__ == "__main__":
    utils.set_soft_gpu(True)
    d = utils.DateData(4000)
    print("Chinese time order: yy/mm/dd ", d.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", d.date_en[:3])
    print("vocabularies: ", d.vocab)
    print("x index sample: \n{}\n{}".format(d.idx2str(d.x[0]), d.x[0]),
          "\ny index sample: \n{}\n{}".format(d.idx2str(d.y[0]), d.y[0]))

    m = Transformer(MODEL_DIM, MAX_LEN, N_LAYER, N_HEAD, d.num_word, DROP_RATE)
    train(m, d, step=800)
    export_attention(m, d)
