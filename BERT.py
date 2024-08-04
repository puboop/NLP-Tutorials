# [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
import numpy as np
import tensorflow as tf
import utils  # this refers to utils.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)
import time
from GPT import GPT
import os
import pickle


class BERT(GPT):
    def __init__(self, model_dim, max_len, n_layer, n_head, n_vocab, lr, max_seg=3, drop_rate=0.1, padding_idx=0):
        super().__init__(model_dim, max_len, n_layer, n_head, n_vocab, lr, max_seg, drop_rate, padding_idx)
        # I think task emb is not necessary for pretraining,
        # because the aim of all tasks is to train a universal sentence embedding
        # the body encoder is the same across all tasks,
        # and different output layer defines different task just like transfer learning.
        # finetuning replaces output layer and leaves the body encoder unchanged.

        # 我认为emb任务不是预训练所必需的，
        # 因为所有任务的目标都是训练一个通用的句子嵌入
        # 身体编码器在所有任务中都是相同的，
        # 不同的输出层定义了不同的任务，就像迁移学习一样。
        # 微调替换了输出层，并保持主体编码器不变。

        # self.task_emb = keras.layers.Embedding(
        #     input_dim=n_task, output_dim=model_dim,  # [n_task, dim]
        #     embeddings_initializer=tf.initializers.RandomNormal(0., 0.01),
        # )

    def step(self, seqs, segs, seqs_, loss_mask, nsp_labels):
        """
        seqs: 输入序列。
        segs: 段信息（表示每个词属于哪个句子）。
        seqs_: 输入序列的真实标签。
        loss_mask: 用于标记哪些位置应该计算损失的掩码。
        nsp_labels: NSP（Next Sentence Prediction）任务的真实标签。
        """
        with tf.GradientTape() as tape:
            mlm_logits, nsp_logits = self.call(seqs, segs, training=True)
            # tf.boolean_mask(..., loss_mask): 使用 loss_mask 过滤掉不需要计算损失的部分。
            # self.cross_entropy(seqs_, mlm_logits): 计算预测 logits 和真实标签之间的交叉熵损失。
            mlm_loss_batch = tf.boolean_mask(self.cross_entropy(seqs_, mlm_logits), loss_mask)
            # 计算 MLM 损失的平均值。
            mlm_loss = tf.reduce_mean(mlm_loss_batch)
            nsp_loss = tf.reduce_mean(self.cross_entropy(nsp_labels, nsp_logits))
            loss = mlm_loss + 0.2 * nsp_loss
            grads = tape.gradient(loss, self.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, mlm_logits

    def mask(self, seqs):
        """
        用于生成掩码，用以标识填充的部分，防止这些部分对模型的训练产生影响
        """
        # tf.math.equal(seqs, self.padding_idx): 对输入序列 seqs 中的每个元素与 padding_idx 进行比较，如果相等则返回 True，否则返回 False。
        # 这一步生成一个布尔张量，形状与 seqs 相同。
        # tf.cast(..., tf.float32): 将布尔张量转换为浮点数张量，其中 True 转换为 1.0，False 转换为 0.0。
        mask = tf.cast(tf.math.equal(seqs, self.padding_idx), tf.float32)
        # 对生成的掩码矩阵进行形状变换，以匹配注意力机制中掩码矩阵的形状要求。
        # mask[:, tf.newaxis, tf.newaxis, :]: 在 mask 的第二个和第三个维度上添加维度，
        # 将掩码矩阵的形状从 [n, step] 变为 [n, 1, 1, step]，其中 n 是批次大小，step 是序列长度。
        return mask[:, tf.newaxis, tf.newaxis, :]  # [n, 1, 1, step]


def _get_loss_mask(len_arange, seq, pad_id):
    """
    len_arange: 一个数组，包含序列中可用于掩码的位置索引。
    seq: 输入序列。
    pad_id: 填充标识符，用于指示掩码位置。

    用于生成一个损失掩码数组。
    这个损失掩码数组用于指示哪些位置需要计算损失，通常用于掩码语言模型 (Masked Language Model, MLM) 的训练。
    具体来说，该函数随机选择输入序列的若干位置进行掩码操作。
    """
    # 使用 np.random.choice 函数从 len_arange 中随机选择若干位置进行掩码操作。
    # size 参数决定了选择位置的数量，max(2, int(MASK_RATE * len(len_arange))) 确保至少选择两个位置，MASK_RATE 是一个预定义的常量，表示掩码比例。
    # replace=False 确保选择的位置不会重复
    rand_id = np.random.choice(len_arange, size=max(2, int(MASK_RATE * len(len_arange))), replace=False)
    # 使用 np.full_like 函数创建一个与 seq 形状相同的数组，并用 pad_id 填充。数据类型为布尔型 (np.bool)。
    loss_mask = np.full_like(seq, pad_id, dtype=np.bool)
    # 将随机选择的位置 (rand_id) 对应的掩码数组位置设置为 True
    loss_mask[rand_id] = True
    return loss_mask[None, :], rand_id


def do_mask(seq, len_arange, pad_id, mask_id):
    """
    seq: 输入序列。
    len_arange: 一个数组，包含序列中可用于掩码的位置索引。
    pad_id: 填充标识符，用于指示掩码位置。
    mask_id: 掩码标识符（如 <MASK>），用于替换选定的位置。

    用于对输入序列中的某些位置进行掩码操作，并返回一个损失掩码数组。
    具体来说，该函数会将选定位置替换为掩码标识符（如 <MASK>），并返回一个指示哪些位置被掩码的布尔数组。
    """
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    # 将输入序列 seq 中 rand_id 对应的位置替换为掩码标识符 mask_id。
    seq[rand_id] = mask_id
    return loss_mask


def do_replace(seq, len_arange, pad_id, word_ids):
    """
    seq: 输入序列。
    len_arange: 一个数组，包含序列中可用于替换的位置索引。
    pad_id: 填充标识符，用于指示替换位置。
    word_ids: 可供替换的词语标识符集合。

    用于将输入序列中的某些位置替换为随机选择的词语，并返回一个损失掩码数组。
    具体来说，该函数会将选定位置替换为随机选择的词语标识符，并返回一个指示哪些位置被替换的布尔数组。
    """
    loss_mask, rand_id = _get_loss_mask(len_arange, seq, pad_id)
    # 将输入序列 seq 中 rand_id 对应的位置替换为从 word_ids 中随机选择的词语标识符。
    seq[rand_id] = np.random.choice(word_ids, size=len(rand_id))
    return loss_mask


def do_nothing(seq, len_arange, pad_id):
    loss_mask, _ = _get_loss_mask(len_arange, seq, pad_id)
    return loss_mask


def random_mask_or_replace(data, arange, batch_size):
    """
    data: 数据对象，包含输入数据及其处理方法。
    arange: 数组，用于指定序列中的索引。
    batch_size: 批次大小

    用于对输入数据进行随机掩码或替换操作，以便用于训练 BERT 模型。
    这种操作被称为 Masked Language Model (MLM)，是 BERT 训练中的一个关键部分

    随机选择一种操作：掩码、保持不变或替换。
    对输入序列的特定位置进行操作，以生成损失掩码。

    """
    # 从 data 中采样一个批次的数据，得到输入序列 seqs、段标识 segs、序列长度 xlen 和下一个句子预测标签 nsp_labels。
    seqs, segs, xlen, nsp_labels = data.sample(batch_size)
    seqs_ = seqs.copy()
    # 生成一个在 [0, 1) 范围内的随机数 p，用于决定将进行哪种操作。
    p = np.random.random()
    # 对每个序列 seqs[i]，在除去 [GO] 和 [SEP] 位置的范围内进行掩码操作。
    # 使用 do_mask 函数将这些位置替换为 [MASK] 标识，并生成损失掩码。
    if p < 0.7:
        # mask
        loss_mask = np.concatenate([do_mask(seqs[i],
                                            # arange 数组中提取特定范围的索引，并拼接这些索引，以便后续用于掩码或替换操作。
                                            # 具体来说，这段代码从 arange 数组中提取两个子数组：一个是从开头到第一个序列长度减去1的位置，
                                            # 另一个是从第一个序列长度加1的位置到两个序列总长度的位置，并将这两个子数组拼接成一个新的数组。

                                            # arange: 一个包含连续整数的数组，通常是用 np.arange 函数生成的。
                                            # xlen: 一个二维数组，每行表示两个子序列的长度。
                                            # i: 当前处理的样本索引。
                                            np.concatenate((arange[:xlen[i, 0]],
                                                            arange[xlen[i, 0] + 1:xlen[i].sum() + 1])),
                                            data.pad_id,
                                            data.v2i["<MASK>"])
                                    for i in range(len(seqs))], axis=0)
    # 对每个序列 seqs[i]，在除去 [GO] 和 [SEP] 位置的范围内保持不变。
    # 使用 do_nothing 函数生成损失掩码。
    elif p < 0.85:
        # do nothing
        loss_mask = np.concatenate([do_nothing(seqs[i],
                                               np.concatenate((arange[:xlen[i, 0]],
                                                               arange[xlen[i, 0] + 1:xlen[i].sum() + 1])),
                                               data.pad_id)
                                    for i in range(len(seqs))], axis=0)
    # 对每个序列 seqs[i]，在除去 [GO] 和 [SEP] 位置的范围内进行替换操作。
    # 使用 do_replace 函数将这些位置替换为随机选择的词汇，并生成损失掩码。
    else:
        # replace
        loss_mask = np.concatenate([do_replace(seqs[i],
                                               np.concatenate((arange[:xlen[i, 0]],
                                                               arange[xlen[i, 0] + 1:xlen[i].sum() + 1])),
                                               data.pad_id,
                                               data.word_ids)
                                    for i in range(len(seqs))], axis=0)
    return seqs, segs, seqs_, loss_mask, xlen, nsp_labels


def train(model, data, step=10000, name="bert"):
    t0 = time.time()
    arange = np.arange(0, data.max_len)
    for t in range(step):
        seqs, segs, seqs_, loss_mask, xlen, nsp_labels = random_mask_or_replace(data, arange, 16)
        loss, pred = model.step(seqs, segs, seqs_, loss_mask, nsp_labels)
        if t % 100 == 0:
            pred = pred[0].numpy().argmax(axis=1)
            t1 = time.time()
            print(
                "\n\nstep: ", t,
                "| time: %.2f" % (t1 - t0),
                "| loss: %.3f" % loss.numpy(),
                "\n| tgt: ", " ".join([data.i2v[i] for i in seqs[0][:xlen[0].sum() + 1]]),
                "\n| prd: ", " ".join([data.i2v[i] for i in pred[:xlen[0].sum() + 1]]),
                "\n| tgt word: ", [data.i2v[i] for i in seqs_[0] * loss_mask[0] if i != data.v2i["<PAD>"]],
                "\n| prd word: ", [data.i2v[i] for i in pred * loss_mask[0] if i != data.v2i["<PAD>"]],
            )
            t0 = t1
    os.makedirs("./visual/models/%s" % name, exist_ok=True)
    model.save_weights("./visual/models/%s/model.ckpt" % name)


def export_attention(model, data, name="bert"):
    model.load_weights("./visual/models/%s/model.ckpt" % name)

    # save attention matrix for visualization
    seqs, segs, xlen, nsp_labels = data.sample(32)
    model.call(seqs, segs, False)
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
    MASK_RATE = 0.15

    d = utils.MRPCData("./MRPC", 2000)
    print("num word: ", d.num_word)
    m = BERT(model_dim=MODEL_DIM,
             max_len=d.max_len,
             n_layer=N_LAYER,
             n_head=4,
             n_vocab=d.num_word,
             lr=LEARNING_RATE,
             max_seg=d.num_seg,
             drop_rate=0.2,
             padding_idx=d.v2i["<PAD>"])
    train(m, d, step=10000, name="bert")
    export_attention(m, d, "bert")
