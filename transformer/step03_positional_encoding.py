import math

import torch
from torch import nn, Tensor


# nn.Transformer does not have positional encoding, you have to code yourself then to add token embeddings.
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, batch_first=False) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if batch_first:
            x = torch.transpose(x,0,1)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        if batch_first:
            return torch.transpose(x,0,1)
        else:
            return x


if __name__ == '__main__':
    pe = PositionalEncoding(4, 0)
    data = torch.zeros(5, 2, 4)  # [seq_len, batch_size, embedding_dim]
    print(pe(data))
    print(pe(data).shape)
    pe = PositionalEncoding(4, 0)
    data = torch.zeros(2, 5, 4)  # [batch_size, seq_len, embedding_dim]
    print(pe(data,batch_first=True))
    print(pe(data,batch_first=True).shape)