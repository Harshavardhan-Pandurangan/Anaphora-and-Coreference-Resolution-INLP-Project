import torch
import torch.nn as nn
from boltons.iterutils import pairwise

from Classes.misc import SpanDist, Speaker, ScoreSeq
from utils import speaker_labels, pad_and_stack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PairwiseScorer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.distance_dim = args.distance_dim
        self.speaker_dim = args.speaker_dim

        self.distance = SpanDist(args)
        self.speaker = Speaker(args)

        self.score = ScoreSeq(args, args.avg_attn_ij_dim)

    def forward(self, spans, scores, states_avg):
        mention_ids, antecedent_ids, distances, speakers = zip(*[(span.id, antecedent.id, span.i-antecedent.j, speaker_labels(span, antecedent)) for span in spans for antecedent in span.yi])
        mention_ids = torch.tensor(mention_ids).to(device)
        antecedent_ids = torch.tensor(antecedent_ids).to(device)

        phi = torch.cat((self.distance(distances), self.speaker(speakers)), dim=1)

        mentions_avg = torch.index_select(states_avg, 0, mention_ids)
        antecedents_avg = torch.index_select(states_avg, 0, antecedent_ids)

        pairs = torch.cat((mentions_avg, antecedents_avg, mentions_avg*antecedents_avg, phi), dim=1)

        mention_scores = torch.index_select(scores, 0, mention_ids)
        antecedent_scores = torch.index_select(scores, 0, antecedent_ids)

        scores = self.score(pairs)

        coref_scores = torch.sum(torch.cat((mention_scores, antecedent_scores, scores), dim=1), dim=1, keepdim=True)

        idxs = [0] + [len(span.yi) for span in spans]
        idxs = [sum(idxs[:i+1]) for i, _ in enumerate(idxs)]
        pairwise_idxs = pairwise(idxs)
        temp_spans = []
        for span, score, (mention_id, antecedent_id) in zip(spans, coref_scores, pairwise_idxs):
            span.yi_idx = [((y.mention_id, y.antecedent_id), (span.mention_id, span.antecedent_id)) for y in span.yi]
            temp_spans.append(span)
        spans = temp_spans

        antecedent_idx = [len(span.yi) for span in spans if len(span.yi)]
        split_scores = [torch.tensor([]).to(device)] + list(torch.split(coref_scores, antecedent_idx, dim=0))

        eps = torch.tensor([[0.]]).to(device).requires_grad_()
        with_eps = [torch.cat((score, eps), dim=0) for score in split_scores]

        probs = [nn.functional.softmax(score) for score in with_eps]
        probs, _ = pad_and_stack(probs, value=1000)
        probs = probs.squeeze()

        return spans, probs, coref_scores