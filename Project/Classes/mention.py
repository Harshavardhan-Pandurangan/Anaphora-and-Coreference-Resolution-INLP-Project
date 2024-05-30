import torch
import torch.nn as nn

from Classes.misc import ScoreSeq, SpanDist
from utils import Span, pad_and_stack, prune

class MentionScorer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.attn = ScoreSeq(args, args.attn_dim)
        self.span = SpanDist(args)
        self.score = ScoreSeq(args, args.avg_attn_i_dim)

    def forward(self, states, embeds, doc, k=250):
        spans = [Span(i, j, id, speaker) for id, i, j, speaker in enumerate(doc)]
        attns = self.attn(states)
        span_attns, span_embeds = zip(*[(attns[span.i:span.j+1], embeds[span.i:span.j+1]) for span in spans])

        padded_attns, _ = pad_and_stack(span_attns, value=-float('inf'))
        padded_embeds, _ = pad_and_stack(span_embeds)

        attn_weights = nn.functional.softmax(padded_attns, dim=1)
        attn_embeds = torch.sum(torch.mul(padded_embeds, attn_weights), dim=1)

        span_len = self.span([len(span) for span in spans])

        states_start_end = torch.stack([torch.cat((states[span.i], states[span.j])) for span in spans])
        states_avg = torch.cat((states_start_end, attn_embeds, span_len), dim=1)

        scores = self.score(states_avg)

        spans_temp = []
        for span, score in zip(spans, scores):
            span.si = score
            spans_temp.append(span)
        spans = spans_temp

        spans = prune(spans, len(doc))

        temp_spans = []
        for idx, span in enumerate(spans):
            span.yi = spans[max(0, idx-k):idx]
            temp_spans.append(span)
        spans = temp_spans

        return spans, scores, states_avg