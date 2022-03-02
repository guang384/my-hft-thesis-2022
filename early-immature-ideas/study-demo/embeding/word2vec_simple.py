# Word2Vec 的 PyTorch 实现（乞丐版） https://wmathor.com/index.php/archives/1443/
# PyTorch 实现 Word2Vec https://wmathor.com/index.php/archives/1435/


# Tensor是n维的数组，在概念上与numpy数组是一样的，不同的是Tensor可以跟踪计算图和计算梯度。

import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 文本预处理
sentences = ["jack like dog", "jack like cat", "jack like animal",
             "dog cat animal", "banana apple cat dog like", "dog fish milk like",
             "dog cat animal like", "jack like apple", "apple like", "jack like banana",
             "apple banana jack movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()  # ['jack', 'like', 'dog', 'jack', 'like', 'cat', 'animal',...]
vocab = list(set(word_sequence))  # build words vocabulary
word2idx = {w: i for i, w in enumerate(vocab)}  # {'jack':0, 'like':1,...}

# Word2Vec Parameters
batch_size = 8
embedding_size = 2  # 2 dim vector represent one word
C = 2  # window size
voc_size = len(vocab)
# 数据预处理
# 1.
skip_grams = []
for idx in range(C, len(word_sequence) - C):
    center = word2idx[word_sequence[idx]]  # center word
    context_idx = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))  # context word idx
    context = [word2idx[word_sequence[i]] for i in context_idx]
    for w in context:
        skip_grams.append([center, w])

'''
 skip_grams: 
  [idx of center, idx of one of context]
'''


# 2.
def make_data(skip_grams):
    input_data = []
    output_data = []
    for i in range(len(skip_grams)):
        input_data.append(np.eye(voc_size)[skip_grams[i][0]])  # np.eye(voc_size) -> one_hot
        output_data.append(skip_grams[i][1])
    return input_data, output_data


# 3.
input_data, output_data = make_data(skip_grams)
input_data, output_data = torch.Tensor(np.array(input_data)), torch.LongTensor(np.array(output_data))
dataset = TensorDataset(input_data, output_data)
loader = DataLoader(dataset, batch_size, shuffle=True)


# Model
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()

        # W and V is not Traspose relationship
        self.W = nn.Parameter(torch.randn(voc_size, embedding_size).type(dtype))
        self.V = nn.Parameter(torch.randn(embedding_size, voc_size).type(dtype))

    def forward(self, X):
        # X : [batch_size, voc_size] one-hot
        # torch.mm only for 2 dim matrix, but torch.matmul can use to any dim
        hidden_layer = torch.matmul(X, self.W)  # hidden_layer : [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.V)  # output_layer : [batch_size, voc_size]
        return output_layer


model = Word2Vec().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(2000):
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        if (epoch) % 1000 == 0:
            print(epoch + 1, i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for i, label in enumerate(vocab):
    W, WT = model.parameters()
    x, y = float(W[i][0]), float(W[i][1])
    print(i, label, x, y)

    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
