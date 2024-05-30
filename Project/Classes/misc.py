import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScoreSeq(nn.Module):
    def __init__(self, args, dim):
        super().__init__()

        self.hidden_dim = args.hidden_dim
        self.embed_dim = dim

        self.score_seq = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout_score),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout_score),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, embeds):
        return self.score_seq(embeds)

class SpanDist(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embed_dim = args.distance_dim

        self.bins = [2**i for i in range(args.bins)]

        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, self.embed_dim),
            nn.Dropout(args.dropout_span)
        )

    def forward(self, args):
        idxs = torch.tensor([sum([True for bin in self.bins if num > bin]) for num in args], requires_grad=False).to(device)
        return self.embeds(idxs)

class Speaker(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embed_dim = args.speaker_dim

        self.embeds = nn.Sequential(
            nn.Embedding(3, self.embed_dim, padding_idx=0),
            nn.Dropout(args.dropout_speaker)
        )

    def forward(self, speakers):
        return self.embeds(torch.tensor(speakers).to(device))