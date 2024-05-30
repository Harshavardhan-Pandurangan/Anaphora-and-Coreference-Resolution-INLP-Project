import torch
import torch.nn as nn
from utils import words_to_idx

class CharCNN(nn.Module):
    def __init__(self, args):
        super().__init__()

        # making the word to index mapping
        self.char_to_idx["<pad>"] = 0
        self.char_to_idx["<unk>"] = 1
        self.char_to_idx = {char: i + 2 for i, char in enumerate(args.char_vocab)}

        self.char_embeds = nn.Embedding(args.char_vocab_size + 2, args.char_embed_dim, padding_idx=0)
        self.conv_list = nn.ModuleList([nn.Conv1d(args.char_embed_dim, args.char_conv_dim, kernel_size=k) for k in [3, 4, 5]])

    def forward(self, words):
        token_embeds = self.char_embeds(words_to_idx(words))
        conv_stack = torch.stack([nn.functional.relu(conv(token_embeds)) for conv in self.conv_list], dim=2)
        pool_stack = nn.functional.max_pool1d(conv_stack, conv_stack.shape[2]).squeeze(2)
        return pool_stack
