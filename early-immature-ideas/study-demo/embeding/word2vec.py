# PyTorch 实现 Word2Vec https://wmathor.com/index.php/archives/1435/

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import scipy
from sklearn.metrics.pairwise import cosine_similarity

SEED = 10086
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

C = 3  # context window
K = 15  # number of negative samples pre positive
epochs = 2
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 100
batch_size = 32
lr = 0.2

# 读取文本
with open('data/text8.train.txt') as f:
    text = f.read()

text = text.lower().split()  # 分割

# 构造 单词到单词出现次数的dict
vocab_dict = dict(
    Counter(text).most_common(MAX_VOCAB_SIZE - 1)
)
vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))  # 没有编号的词全部统计到UNK上

# 映射关系

word2idx = {word: i for i, word in enumerate(vocab_dict.keys())}
idx2word = {i: word for i, word in enumerate(vocab_dict.keys())}

word_counts = np.array([count for count in vocab_dict.values()], dtype=float)
word_freqs = word_counts / np.sum(word_counts)  # 词频

word_freqs = word_freqs ** (3. / 4)  # 论文里要求频率变为原来的0.75次方


# DataLoader
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self,
                 text,
                 word2idx,
                 word_freqs):
        '''
        text: 拆分成单词的语料
        word2idx: 单词到索引号（编号）的映射
        word_freqs: 词频
        '''
        super(WordEmbeddingDataset, self).__init__()

        # 编码后的语料
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]  # 把语料库用索引号编码
        self.text_encoded = torch.LongTensor(self.text_encoded)

        # 单词索引
        self.word2idx = word2idx
        # 词频
        self.word_freqs = torch.Tensor(word_freqs)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        '''
        返回：
        - 中心词
        - 相关词 positive word
        - 随机采样得到的 negative word  ( K倍的positive word )
        '''
        center_word = self.text_encoded[idx]

        positive_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        positive_indices = [i % len(self.text_encoded) for i in positive_indices]  # 避免越界，取个余数

        positive_words = self.text_encoded[positive_indices]  # positive_indices是个列表，返回的会是个tensor

        negative_words = torch.multinomial(self.word_freqs, K * positive_words.shape[0], True)  # 多项式分布抽样 K倍正向词数量

        # wihle循环确保正负向词无交集
        while len(
                set(positive_words.numpy().tolist()) & set(negative_words.numpy().tolist())  # 取交集
        ) > 0:
            negative_words = torch.multinomial(self.word_freqs, K * positive_words.shape[0], True)

        return center_word, positive_words, negative_words


dataset = WordEmbeddingDataset(text, word2idx, word_freqs)
dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)


# print(next(iter(dataset)))  # 验证数据集可用


# 定义模型
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.input_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.output_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, positive_labels, negative_labels):
        '''
        input_labels: center words,    shape:  [batch_size]
        positive_labels: positive words,  shape: [batch_size, context window size * 2]   乘以二是因为有前文后文两部分
        negative_labels: negative words, shape: [batch_size, K * context window size * 2]

        return loss, shape:[batch_size]
        '''
        input_embedding = self.input_embed(input_labels)  # shape: [batch_size,embed size]
        positive_embedding = self.output_embed(positive_labels)  # [batch_size, (window * 2), embed_size]
        negative_embedding = self.output_embed(negative_labels)  # [batch_size, (window * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]

        # bmm(a, b)，batch matrix multiply
        positive_dot = torch.bmm(positive_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        negative_dot = torch.bmm(negative_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]

        positive_dot = positive_dot.squeeze(2)  # [batch_size, (window * 2)]
        negative_dot = negative_dot.squeeze(2)  # batch_size, (window * 2 * K)]

        log_positive = F.logsigmoid(positive_dot).sum(1)
        log_negative = F.logsigmoid(negative_dot).sum(1)

        loss = log_positive + log_negative

        return -loss

    def input_embedding(self):
        return self.input_embed.weight.detach().numpy()


model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练

for e in range(epochs):
    for i, (input_labels, positivate_labels, negative_lables) in enumerate(dataloader):
        input_labels = input_labels.long()
        positivate_labels = positivate_labels.long()
        negative_lables = negative_lables.long()

        optimizer.zero_grad()
        loss = model(input_labels, positivate_labels, negative_lables).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('epoch', e, 'iteration', i, loss.item())

embedding_weights = model.input_embedding()
torch.save(model.state_dict(), "")
torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))


# 验证
def find_nearest(word):
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx2word[i] for i in cos_dis.argsort()[:10]]


for word in ["two", "america", "computer"]:
    print(word, find_nearest(word))
