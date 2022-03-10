import pandas as pd
import csv
import html

import torch
import torch.utils.data as tud
from torch.nn.utils.rnn import pad_sequence

from tokenizers.implementations import BertWordPieceTokenizer


def load_raw_data_to_corpus(pt_corpus_file_path, en_corpus_file_path):
    vocab_en = pd.read_csv(en_corpus_file_path, delimiter='\t',
                           encoding='utf-8', quoting=csv.QUOTE_NONE, header=None)
    vocab_pt = pd.read_csv(pt_corpus_file_path, delimiter='\t',
                           encoding='utf-8', quoting=csv.QUOTE_NONE, header=None)

    # 语句中有被转义的单引号等
    vocab_en[0] = vocab_en[0].apply(html.unescape)
    vocab_pt[0] = vocab_pt[0].apply(html.unescape)
    # 葡萄牙语都以 __en__ 开头，把这个去掉
    vocab_pt[0] = vocab_pt[0].apply(lambda x: x[7:])

    # 两个Series组合成员给Dataframe
    dataframe = pd.DataFrame()
    dataframe['pt'] = vocab_pt[0]
    dataframe['en'] = vocab_en[0]
    # 返回结果
    return dataframe


def load_tokenizer(tokenizer_model_file_path):
    loaded = BertWordPieceTokenizer(tokenizer_model_file_path,
                                    clean_text=True,
                                    handle_chinese_chars=True,
                                    strip_accents=False,
                                    lowercase=False)  # 参数配置应该和训练时候一致否则编码结果不一样~
    return loaded


def data_encode(tokenizer, vocabs_dataframe, if_plot):
    ret = pd.DataFrame()
    for col in vocabs_dataframe.columns:
        encoded_batch = tokenizer.encode_batch(list(vocabs_dataframe[col]))

        if if_plot:
            lens = []
            for encoded_data in encoded_batch:
                lens.append(len(encoded_data))
            import matplotlib.pyplot as plt
            plt.plot(lens)
            plt.suptitle(col)
            plt.show()
        col_tokens = []
        col_ids = []
        for encoded_data in encoded_batch:
            col_ids.append(encoded_data.ids)
            col_tokens.append(encoded_data.tokens)
        ret[col + '_ids'] = pd.Series(col_ids)
        ret[col + '_tokens'] = pd.Series(col_tokens)
    return ret


def data_filter(df, length=128):
    df['en_len'] = df['en_tokens'].apply(len)
    df['pt_len'] = df['pt_tokens'].apply(len)
    df = df[(df['en_len'] < length) & (df['pt_len'] < length)]
    del df['en_len']
    del df['pt_len']
    return df


# 补0,并转为tensor
def padding_and_to_tensor(series, width):
    # 补0
    ret = pad_sequence(list(series.apply(torch.IntTensor)),
                       batch_first=True)
    # 检查宽度够不够
    if ret.shape[1] < width:
        # If a 4-tuple, uses padding_left, padding_right, padding_top, padding_bottom
        pad_right = torch.nn.ZeroPad2d(padding=(0, width - ret.shape[1], 0, 0))
        ret = pad_right(ret)
    return ret


# DataLoader
class Pt2enCorpusDataset(tud.Dataset):
    def __init__(self,
                 pt_corpus_file_path,
                 en_corpus_file_path,
                 tokenizer_model_file_path,
                 embedding_width=40):
        """
        pt_corpus_file_path: 葡萄牙语料库
        en_corpus_file_path: 对应的英文语料库
        tokenizer_model_file_path: 训练好的字词分词器模型地址
        """
        super(Pt2enCorpusDataset, self).__init__()

        # 加载语料
        corpus_df = load_raw_data_to_corpus('data/corpus_pt_en/train.pt', 'data/corpus_pt_en/train.en')
        # 加载分词器
        self.tokenizer = load_tokenizer('data/tokenizer_model/bert-wordpiece-train-25000-vocab.txt')
        # 编码数据
        encoded_df = data_encode(self.tokenizer, corpus_df, if_plot=False)
        # 过滤数据（保留长度在指定编码长度以内的）
        encoded_df = data_filter(encoded_df, length=embedding_width)
        # 补0 , 即用[PAD]补齐拆开的句子
        self.pt_tensor = padding_and_to_tensor(encoded_df['pt_ids'], width=embedding_width)
        self.en_tensor = padding_and_to_tensor(encoded_df['en_ids'], width=embedding_width)

    def __len__(self):
        return len(self.pt_tensor)

    def get_vocab_size(self):  # 词库词汇量
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def __getitem__(self, idx):
        """
        返回：
        一对编码好的葡萄牙语以及对应的英语编码
        """
        return self.pt_tensor[idx], self.en_tensor[idx]

    def decode(self, tokens):
        return self.tokenizer.decode(tokens.numpy())


if __name__ == '__main__':
    # 加载语料
    corpus = load_raw_data_to_corpus('data/corpus_pt_en/train.pt', 'data/corpus_pt_en/train.en')
    print('\n--- ROW CORPUS: ---\n', corpus[['pt', 'en']])
    # 加载分词器
    the_tokenizer = load_tokenizer('data/tokenizer_model/bert-wordpiece-train-25000-vocab.txt')
    # 编码数据
    encoded = data_encode(the_tokenizer, corpus, if_plot=False)
    print('\n--- ENCODED CORPUS OF PT: ---\n', encoded[['pt_ids', 'pt_tokens']])
    print('\n--- ENCODED CORPUS OF EN: ---\n', encoded[['en_ids', 'en_tokens']])
    # 过滤数据（保留长度在40以内的）
    print('\n--- BEFORE FILTER : ---\n', len(encoded))
    encoded = data_filter(encoded, length=40)
    print('\n--- AFTER  FILTER : ---\n', len(encoded))
    # 补0 , 即用[PAD]补齐拆开的句子
    pt_tensor = padding_and_to_tensor(encoded['pt_ids'], width=40)
    en_tensor = padding_and_to_tensor(encoded['en_ids'], width=40)
    print('\n--- PADDED CORPUS OF PT: ---\n', pt_tensor)
    print('\n--- PADDED CORPUS OF EN: ---\n', en_tensor)
    # 验证
    print('\n--- CHECK DECODE: ---\n')
    print('pt_tensor:', the_tokenizer.decode(pt_tensor[3].numpy()))
    print('pt_corpus:', corpus['pt'].iloc[4])  # 因为过滤了长语料 所以序号不一致
    print('en_tensor:', the_tokenizer.decode(en_tensor[3].numpy()))
    print('en_corpus:', corpus['en'].iloc[4])

    # 数据集
    dataset = Pt2enCorpusDataset(pt_corpus_file_path='data/corpus_pt_en/train.pt',
                                 en_corpus_file_path='data/corpus_pt_en/train.en',
                                 tokenizer_model_file_path='data/tokenizer_model/bert-wordpiece-train-25000-vocab.txt',
                                 embedding_width=64)
    dataloader = tud.DataLoader(dataset, 12, shuffle=True, num_workers=1)

    print('dataset size', len(dataset))
    print('dataloader batch count', len(dataloader))
    one_pt_batch, one_en_batch = next(iter(dataloader))
    print(one_pt_batch, one_en_batch)
    one_pt, one_en = one_pt_batch[0], one_en_batch[0]

    print('pt-> ', dataset.decode(one_pt))
    print('en-> ', dataset.decode(one_en))
    print('vocab_size-> ', dataset.get_vocab_size())
