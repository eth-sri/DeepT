import torch

vocab_word_to_idx = torch.load('vocab.data')
vocab_idx_to_word = {index: word for (word, index) in vocab_word_to_idx.items()}

keys = list(sorted(vocab_idx_to_word.keys()))
assert keys == list(range(len(keys)))

with open('vocab_base.txt', 'w') as f:
    for key in keys:
        word = vocab_idx_to_word[key]
        f.write(f"{word}\n")
