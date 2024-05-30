import torch
import torch.nn as nn
from Classes.token_rel_embeds import CharCNN
import gensim.downloader as api
from utils import tokens_to_idx

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        glove_embeds = api.load("word2vec-google-news-300")
        turian_embeds = api.load("fasttext-wiki-news-subwords-300")

        self.glove_embeds = nn.Embedding(len(glove_embeds.index_to_key), 300)
        self.glove_embeds.weight.data.copy_(torch.tensor(glove_embeds.vectors))
        self.glove_embeds.weight.requires_grad = False

        self.turian_embeds = nn.Embedding(len(turian_embeds.index_to_key), 300)
        self.turian_embeds.weight.data.copy_(torch.tensor(turian_embeds.vectors))
        self.turian_embeds.weight.requires_grad = False

        self.char_embeds = CharCNN(args)

        self.bi_lstm = nn.LSTM(self.glove_embeds.embedding_dim + self.turian_embeds.embedding_dim + args.char_conv_dim, args.hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout_embeds = nn.Dropout(args.dropout_embeds, inplace=True)
        self.dropout_lstm = nn.Dropout(args.dropout_lstm, inplace=True)

    def forward(self, sentences):
        glove_embeds = self.glove_embeds(tokens_to_idx(sentences, self.glove_embeds, self.glove_embeds.index_to_key))
        turian_embeds = self.turian_embeds(tokens_to_idx(sentences, self.turian_embeds, self.turian_embeds.index_to_key))
        char_embeds = self.char_embeds(sentences)
        embeds = torch.cat([glove_embeds, turian_embeds, char_embeds], dim=1)

        lens = [len(embed) for embed in embeds]
        sorted_lens = sorted(range(len(lens)), key=lambda i: lens[i], reverse=True)
        sorted_embeds = embeds[sorted_lens]
        packed_embeds = nn.utils.rnn.pack_sequence(sorted_embeds)

        dropout_embeds = self.dropout_embeds(packed_embeds.data)
        lstm_out, _ = self.bi_lstm(dropout_embeds)
        dropout_lstm = self.dropout_lstm(lstm_out)

        unpacked, _ = nn.utils.rnn.pad_packed_sequence(dropout_lstm)
        unpadded = [unpacked[idx][:lens[sorted_lens[idx]]] for idx in range(len(lens))]
        regrouped = [unpadded[sorted_lens.index(idx)] for idx in range(len(lens))]

        return torch.cat(regrouped, dim=0), torch.cat(embeds, dim=0)
