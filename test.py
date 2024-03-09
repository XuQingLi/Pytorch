import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

line=1
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)
line=1
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)
line=1
def make_data(sentences):
    # 初始化编码器输入、解码器输入和解码器输出列表
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        # 将句子中的单词转换为对应的索引
        enc_input = [src_vocab[n] for n in sentences[i][0].split()]  # e.g., [1, 2, 3, 4, 0]
        dec_input = [tgt_vocab[n] for n in sentences[i][1].split()]  # e.g., [6, 1, 2, 3, 4, 8]
        dec_output = [tgt_vocab[n] for n in sentences[i][2].split()] # e.g., [1, 2, 3, 4, 8, 7]

        # 将索引列表添加到对应的总列表中
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    # 将列表转换为PyTorch的LongTensor返回
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
line=1