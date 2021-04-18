import torch

from Models.utils import InputFeatures

word_embeddings = torch.load("word_embeddings.data")
vocab = torch.load("vocab.data")
config = torch.load("model.config")







def get_input(batch):
    label_list = range(0, 2)
    device = 'cpu'

    features = convert_examples_to_features(batch, label_list, 100000000, vocab, drop_unk=False)

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(device)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(device)
    tokens = [f.tokens for f in features]

    embeddings = word_embeddings(input_ids)

    return embeddings, tokens, label_ids


e, t, l = get_input([{"sentence":"hello world", "label": 0}])
