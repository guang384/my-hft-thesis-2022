# 任务说明

<font color=#008000 size=5>
用 Transformer 训练一个由[葡萄牙语]至[英语]的翻译器。
</font>

## 数据集说明

TED演讲数据集

Dataset : ted_hrlr_translate

Description:
Data sets derived from TED talk transcripts for comparing similar language pairs where one is high resource and the other is low resource.

Homepage: https://github.com/neulab/word-embeddings-for-nmt

Download size: 124.94 MiB

---
## 数据集获取
有这么几个途径，大同小异。
###途径一：

1.下载数据 http://phontron.com/data/ted_talks.tar.gz

2.解压成三个TSV文件(用tab分割的csv)

3.用官方 [GIT代码中的reader](https://raw.githubusercontent.com/neulab/word-embeddings-for-nmt/master/ted_reader.py) 读取下载好的数据抽取出葡萄牙语到英语的数据集

这个reader执行的时候需要改两个地方 就是读取和写入的时候改成用 utf8 和text模式读取和写入：
```
 with open(path, 'w', encoding="utf-8") as fp:
    # some code
    
 with open(path, 'rt', encoding="utf-8") as fp:
    # some code
```

数据整理到DataFrame里查看下大概这样：

```
                                                     pt                                                 en
0     Isso corresponde ao dobro do tempo da existênc...  That 's twice as long as humans have been on t...
1     Onde é que se encontram estas condições Goldil...  Now , where do you find such Goldilocks condit...
2     Mas , evidentemente , a vida é mais do que mer...  But of course , life is more than just exotic ...
3                       Cada troço contém informações .                   Each rung contains information .
4                             Eu não era uma ativista .                            I was not an activist .
...                                                 ...                                                ...
1188  A continuidade da humanidade , com certeza , m...  The continuum of humanity , sure , but in a la...
1189  E nós temos uma escolha a fazer durante nossa ...  And we have a choice to make during our brief ...
1190                   Para vocês , é a vossa decisão .                        For you , it 's your call .
1191                                         Obrigado .                                        Thank you .
1192                                       ( Aplausos )                                       ( Applause )

[1193 rows x 2 columns]

```
###途径二：
也可以 [直接下载原始数据](http://www.phontron.com/data/qi18naacl-dataset.tar.gz) 自行解析成为需要的格式。
###途径三：
还可以通过tensorflow_dataset下载，只是下载较慢（好在数据集本身没多大）
```python
import tensorflow_datasets as tfds   # 用于数据集 (缺点 依赖tensorflow

# 下载数据集
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True,data_dir='data')
train_examples, val_examples =examples['train'], examples['validation']
```
---
## 词嵌入

### Tensorflow SubwordTextEncoder 子词分词器
网上很多例子是用这个分词器分词的。但是这个移动到deprecated包下了...

移到deprecated包的理由 [看起来](https://github.com/tensorflow/datasets/issues/2879) 主要因为没人维护。

```python
import tensorflow_datasets as tfds 

train_examples = ...  #加载的数据集 

tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))
# Tokenized string is [7915, 1248, 7946, 7194, 13, 2799, 7877]

original_string = tokenizer_en.decode(tokenized_string)
print ('The original string: {}'.format(original_string))
assert original_string == sample_string

```

子词分词方法主要有三种：

- BPE : Byte Pair Encoding (Sennrich et al., 2015)
- WordPiece : (Schuster et al., 2012)
- ULM : Unigram Language Model (Kudo, 2018)

BERT用了WordPiece,所以这里也用WordPiece做子词分词算法
（Tensorflow 的 SubwordTextEncoder 算法从源码上看起来像...BPE?）

from **torchtext.vocab** import **build_vocab_from_iterator** 这个貌似是以单词为单位编码，统计词频然后进行编码，不是子词分词方法

Anyway, 最后决定用 [huggingface提供的**tokenizers**库](https://github.com/huggingface/tokenizers) 的 BertWordPieceTokenizer [做子分词](https://github.com/huggingface/tokenizers/blob/master/bindings/python/examples/train_bert_wordpiece.py)

## 位置编码
查看源码和网上查询确认，torch的 Transformer Model 不带，位置编码。

在他们关于transformer的[示例教程](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) 中有提到过位置编码（但是没有加入torch.nn 中）
所以这里直接复制了这部分代码

```python
import math
import torch
from torch import nn, Tensor

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

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
```