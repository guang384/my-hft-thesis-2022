import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from step03_positional_encoding import PositionalEncoding

# https://blog.csdn.net/SangrealLilith/article/details/103527408

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

dropout_rate = 0.1


class Translator(nn.Module):
    def __init__(self,
                 vocab_size,  # 词汇表大小
                 position_encoding_dropout_rate,
                 transformer_d_model,
                 transformer_num_heads,
                 transformer_num_layers,
                 transformer_dim_feedforward,
                 transformer_dropout_rate
                 ):
        super().__init__()
        self.d_model = transformer_d_model
        self.embedding = nn.Embedding(vocab_size, transformer_d_model)  # 词汇表是按照数字顺序编码的，这里需要将”编号“嵌入成”向量“表示的词嵌入
        self.position_encoding = PositionalEncoding(transformer_d_model, position_encoding_dropout_rate)  # 编码位置
        self.transformer = nn.Transformer(d_model=transformer_d_model,
                                          nhead=transformer_num_heads,
                                          num_encoder_layers=transformer_num_layers,
                                          num_decoder_layers=transformer_num_layers,
                                          dim_feedforward=transformer_dim_feedforward,
                                          dropout=transformer_dropout_rate)

    def forward(self,
                tokenized_src: Tensor,
                tokenized_tgt: Tensor,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len , embedding_dim]

        编码过程
            根据输入创建mask （Lookahead Mask）
            将原始句子单词编码更换为词嵌入
            (注意：在原始句子的词嵌入上需要乘上d_model的平方根 ，这是原论文中规定的。文章中没有对这个变量的解释。)
            将词嵌入加上位置编码
            加上MASK
            丢到Transformer中


        """
        # 准备mask
        src_mask = tokenized_src == 0
        tgt_mask = tokenized_tgt == 0

        # TODO : 找到了更好的例子 这个暂时不搞了

        # 转成词嵌入
        embed_tokenized_src = self.embedding(tokenized_src)
        embed_tokenized_tgt = self.embedding(tokenized_tgt)
        embed_tokenized_src *= math.sqrt(self.d_model)  # 原论文中规定 词嵌入需要乘上d_model的平方根
        embed_tokenized_tgt *= math.sqrt(self.d_model)  # 原论文中规定 词嵌入需要乘上d_model的平方根

        # 加上位置编码
        src = self.position_encoding(embed_tokenized_src, batch_first=True)
        tgt = self.position_encoding(embed_tokenized_tgt, batch_first=True)

        self.transformer()
        nn.TransformerEncoderLayer

    def _generate_lookahead_mask(self, sz: int) -> Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask


model = nn.Transformer(d_model=d_model,
                       nhead=num_heads,
                       num_encoder_layers=num_layers,
                       num_decoder_layers=num_layers,
                       dim_feedforward=dff,
                       dropout=dropout_rate)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

# https://www.analyticsvidhya.com/blog/2021/06/language-translation-with-transformer-in-python/
