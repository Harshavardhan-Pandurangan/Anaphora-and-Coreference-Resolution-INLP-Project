import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stack_tokens(tokens, max_len):
    tokens = [t[:max_len] for t in tokens]
    tokens = [torch.cat([t, torch.zeros(max_len - t.size(0)).long().to(device)]) for t in tokens]
    return torch.stack(tokens)

def word_to_idx(word, char_to_idx):
    return torch.tensor([char_to_idx.get(char, 1) for char in word]).to(device)

def words_to_idx(words, char_to_idx, max_len):
    tokens = [word_to_idx(word, char_to_idx) for word in words]
    embeds = stack_tokens(tokens, max_len)
    return embeds

def tokens_to_idx(sentences, embeds, embeds_index_to_key):
    return torch.tensor([[embeds_index_to_key.index(token) for token in sentence] for sentence in sentences]).to(device)

class Span:
    def __init__(self, i, j, id, speaker):
        self.i = i
        self.j = j
        self.id = id
        self.speaker = speaker
        self.si = None
        self.yi = None
        self.yi_idx = None

    def __len__(self):
        return self.j - self.i + 1

def pad_and_stack(tensors, pad_size=None, value=0):
    lens = [t.shape[0] for t in tensors]

    if not pad_size:
        pad_size = max(lens)

    padded = torch.stack([nn.functional.pad(input=t[:pad_size], pad=(0, 0, 0, max(0, pad_size - size)), value=value) for t, size in zip(tensors, lens)], dim=0)
    return padded, lens

def prune(spans, T, Lambda=0.4):
    stop = int(T * Lambda)
    spans = sorted(spans, key=lambda x: x.si, reverse=True)
    non_overlapping = []
    seen = set()
    for span in spans:
        idxs = range(span.i, span.j+1)
        taken = [i in seen for i in idxs]
        if len(set(taken)) == 1 or (taken[0] == taken[-1] == False):
            non_overlapping.append(span)
            seen.update(idxs)

    pruned_spans = non_overlapping[:stop]

    spans = sorted(pruned_spans, key=lambda x: (x.i, x.j))
    return spans

def speaker_labels(span, antecedent):
    if span.speaker == antecedent.speaker:
        return torch.tensor(1).to(device)
    elif span.speaker != antecedent.speaker:
        return torch.tensor(2).to(device)
    else:
        return torch.tensor(0).to(device)
