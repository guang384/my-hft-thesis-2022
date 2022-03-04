# PyTorch 实现 Word2Vec https://wmathor.com/index.php/archives/1435/
import timeit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import scipy
from sklearn.metrics.pairwise import cosine_similarity

C = 3  # context window
K = 15  # number of negative samples pre positive
EPOCHS = 1
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 100

# 调整BATCH_SIZE和DATALOADER_WORKER_NUMS让数据集产生和消费达到一个平衡 效果最佳
BATCH_SIZE = 16  # 每个次迭代从数据集中取出的数据多少，决定迭代轮次，另外考虑dataloader加载效率
DATALOADER_WORKER_NUMS = 4  # 本方案数据集采样CPU密集型，GPU加成不足，采用多个worker提高效率。dataloader的并行机制是生产者消费者模式？

LR = 1e-3
SEED = 10086

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 读取文本 拆分并按照单词索引
def encoder_word(text):
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

    return word2idx, idx2word, word_freqs


# DataLoader
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self,
                 text,
                 word2idx,
                 word_freqs):
        """
        text: 拆分成单词的语料
        word2idx: 单词到索引号（编号）的映射
        word_freqs: 词频
        """
        super(WordEmbeddingDataset, self).__init__()

        # 编码后的语料
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]  # 把语料库用索引号编码
        self.text_encoded = torch.LongTensor(self.text_encoded)

        # 单词索引
        self.word2idx = word2idx
        # 词频
        scaled_word_freqs = word_freqs ** (3. / 4)  # 论文里要求频率变为原来的0.75次方 , 猜测是非均衡缩放下频率这样抽样会稍微平均一点
        self.word_freqs = torch.Tensor(scaled_word_freqs)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        """
        返回：
        - 中心词
        - 相关词 positive word
        - 随机采样得到的 negative word  ( K倍的positive word )
        """
        center_word = self.text_encoded[idx]

        positive_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        positive_indices = [i % len(self.text_encoded) for i in positive_indices]  # 避免越界，取个余数

        positive_words = self.text_encoded[positive_indices]  # positive_indices是个列表，返回的会是个tensor

        # 多项式分布抽样 K倍正向词数量
        negative_words = torch.multinomial(self.word_freqs, K * positive_words.shape[0], True)

        # While 循环确保正负向词无交集(GPU下用tensor算交集
        while len(
                set(negative_words.cpu().numpy().tolist()) & set(positive_words.cpu().numpy().tolist())  # 取交集
        ) > 0:
            negative_words = torch.multinomial(self.word_freqs, K * positive_words.shape[0], True)

        return center_word, positive_words, negative_words


# 定义模型
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.input_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.output_embed = nn.Embedding(self.vocab_size, self.embed_size)

        self.to(device)

    def forward(self, input_labels, positive_labels, negative_labels):
        """
        input_labels: center words,    shape:  [batch_size]
        positive_labels: positive words,  shape: [batch_size, context window size * 2]   乘以二是因为有前文后文两部分
        negative_labels: negative words, shape: [batch_size, K * context window size * 2]

        return loss, shape:[batch_size]
        """
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
        return self.input_embed.weight.detach().cpu().numpy()


# 训练
def train(text, word2idx, word_freqs):
    dataset = WordEmbeddingDataset(text, word2idx, word_freqs)
    dataloader = tud.DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKER_NUMS)
    print('dataset word size: ', len(dataset))

    model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
    print('model on cuda : ', next(model.parameters()).is_cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for e in range(EPOCHS):
        print('EPOCH', EPOCHS, 'STARTED !')
        epoch_start = timeit.default_timer()
        calc_total_time = 0
        for i, (input_labels, positive_labels, negative_labels) in enumerate(dataloader):
            start = timeit.default_timer()
            input_labels = input_labels.long().to(device)
            positive_labels = positive_labels.long().to(device)
            negative_labels = negative_labels.long().to(device)

            optimizer.zero_grad()
            loss = model(input_labels, positive_labels, negative_labels).mean()
            loss.backward()
            optimizer.step()
            calc_use = timeit.default_timer() - start
            calc_total_time += calc_use
            if i % 100 == 0 and i > 0:
                used_time = timeit.default_timer() - epoch_start
                print(
                    # 轮次
                    'EPOCH: ', e + 1, '/', EPOCHS + 1,
                    # 迭代进度
                    '| ITER: ', i, '/', len(dataloader),
                    # 迭代一次用于训练计算的时间和迭代一次用时（其他时间因为数据集采样消耗
                    '| TRAINING TIME PER ITER:', round(calc_total_time / i, 4), '/', round(used_time / i, 4),
                    # 估计剩余时间
                    ' , ETA :', round((used_time / i) * (len(dataloader) - i), 2),
                    # 误差
                    '| LOSS: ', loss.item(), )
    print('Training done.')
    return model


# 验证
def test(word2idx, idx2word, embedding_weights):
    def find_nearest(word):
        index = word2idx[word]
        embedding = embedding_weights[index]
        cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
        return [idx2word[i] for i in cos_dis.argsort()[:10]]

    for w in ["two", "america", "computer"]:
        print(w, find_nearest(w))
    """
    OUTPUT >>
    
    two ['two', 'three', 'four', 'five', 'zero', 'six', 'seven', 'one', 'eight', 'nine']
    america ['america', 'europe', 'africa', 'asia', 'australia', 'north', 'south', 'canada', 'korea', 'india']
    computer ['computer', 'computers', 'video', 'hardware', 'software', 'digital', 'computing', 'graphics', 'engineering', 'technology']
    
    """


if __name__ == '__main__':
    print('run on device : ', device)

    DATA_SET = 'dev'

    # 读取文本
    with open('data/text8.dev.txt') as f:
        the_text = f.read()

    # 分词
    the_text = the_text.lower().split()
    print('Got', len(the_text), 'words of text.')

    # 编码单词
    the_word2idx, the_idx2word, the_word_freqs = encoder_word(the_text)

    # save word encoded
    torch.save((the_word2idx, the_idx2word, the_word_freqs),
               "mapping_dataset-{}_embedding-size-{}.th".format(DATA_SET, EMBEDDING_SIZE))
    print('encoded word saved!')

    # 加载 编码好的单词
    (the_word2idx, the_idx2word, the_word_freqs) = torch.load(
        "mapping_dataset-{}_embedding-size-{}.th".format(DATA_SET, EMBEDDING_SIZE))
    print('encoded word loaded!')

    '''
    训练
    '''
    trained_model = train(text=the_text, word2idx=the_word2idx, word_freqs=the_word_freqs)

    # 保存训练好的 embedding
    embedding_weights = trained_model.input_embedding()
    torch.save(trained_model.state_dict(), "embedding_dataset-{}_embedding-size-{}.th".format(DATA_SET, EMBEDDING_SIZE))
    print('model saved!')

    # 加载训练好的 embedding
    embedding_weights = torch.load("embedding_dataset-{}_embedding-size-{}.th".format(DATA_SET, EMBEDDING_SIZE))
    print('model loaded!')

    '''
    测试
    '''
    test(the_word2idx, the_idx2word, embedding_weights)
