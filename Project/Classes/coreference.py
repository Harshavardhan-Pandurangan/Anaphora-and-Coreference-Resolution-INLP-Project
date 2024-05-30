import torch
import torch.nn as nn

from Classes.encoder import Encoder
from Classes.mention import MentionScorer
from Classes.pairwise import PairwiseScorer

class CoreferenceResolution(nn.Module):
    def __init__(self, args):
        super().__init__()

        attn_dim = args.hidden_dim * 2
        avg_attn_i_dim = attn_dim * 2 + args.embed_dim + args.distance_dim
        avg_attn_ij_dim = avg_attn_i_dim * 3 + args.distance_dim  + args.speaker_dim

        args.attn_dim = attn_dim
        args.avg_attn_i_dim = avg_attn_i_dim
        args.avg_attn_ij_dim = avg_attn_ij_dim

        self.encoder = Encoder(args)
        self.score_spans = MentionScorer(args)
        self.score_pairs = PairwiseScorer(args)

    def forward(self, doc):
        states, embeds = self.encoder(doc)
        spans, scores, states_avg = self.score_spans(states, embeds, doc)
        spans, probs = self.score_pairs(spans, scores, states_avg)
        return spans, probs