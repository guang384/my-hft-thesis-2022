from tokenizers.implementations import BertWordPieceTokenizer

# Initialize an empty tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False,
    lowercase=False,
)
# And then train
tokenizer.train(
    ['data/corpus_pt_en/train.en', 'data/corpus_pt_en/train.pt'],
    vocab_size=25000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    wordpieces_prefix="##",
)

# Save the files
tokenizer.save_model('data/tokenizer_model', 'bert-wordpiece-train-25000')
print(tokenizer.encode("Para vocês , é a vossa decisão").ids)
print(tokenizer.encode("Para vocês , é a vossa decisão").tokens)

print(tokenizer.decode(tokenizer.encode("Hello world,this is a World").ids))

for tid in tokenizer.encode("Para vocês , é a vossa decisão").ids:
    print(tokenizer.id_to_token(tid))

# 加载
loaded = BertWordPieceTokenizer('data/tokenizer_model/bert-wordpiece-25000-vocab.txt',
                                clean_text=True,
                                handle_chinese_chars=True,
                                strip_accents=False,
                                lowercase=False)  # 参数配置应该和训练时候一致否则编码结果不一样~

print(loaded.encode("Para vocês , é a vossa decisão"))
print(loaded.encode("Para vocês , é a vossa decisão").ids)
print(loaded.encode("Para vocês , é a vossa decisão").tokens)
