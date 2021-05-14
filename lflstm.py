"""
forward(token_idx_seqs, var_idx_seqs):
    #   (batch size, max len, emb dim)
    token_embs = self.embeddings(token_idx_seqs)

    #   (batch size, max len, var emb dim)
    var_embs = make_var_embs(var_idx_seqs)

    embs = token_embs + var_embs

    #   (batch size, repr dim)
    return something
"""

import re
import sys

import torch
from torch import nn
from torch.distributions.normal import Normal

from lf_tokenizer import LFTokenizer

GLOVE_PATH = "/usr1/data/sozaki/glove.42B.300d.txt"

def update_embs_with_glove(tokenizer, embs):
    my_sym2idx = dict()

    #   sym2idx from tokenizer, with leading underscores stripped
    for k, v in tokenizer.sym2idx.items():
        newk = re.sub("_dash_", "-", k[1:])
        my_sym2idx[newk] = v

    num_found = 0

    print(f"Loading GloVe embeddings from {GLOVE_PATH}...", file=sys.stderr)

    with open(GLOVE_PATH, "r") as f:
        for line in f:
            splitt = line.split()

            word = splitt[0]
            rest = splitt[1:]

            if word.lower() in my_sym2idx:
                rest = list(map(lambda a: float(a), rest))
                idx = my_sym2idx[word]
                embs[idx] = torch.tensor(rest)

                num_found += 1

    num_syms = len(my_sym2idx)
    percent = num_found * 10000 // num_syms / 100

    print(f"{num_found}/{num_syms} ({percent}%) of symbols matched from GloVe.", file=sys.stderr)

    return embs

class LFModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        token_emb_init="random",
        token_emb_dim=300,
        var_emb_dim=50,
        hidden_size=512,
        num_layers=1,
        dropout=0.,
    ):
        super(LFModel, self).__init__()

        num_embs = len(tokenizer.idx2sym)

        dist = Normal(0., 0.001)

        token_embs = dist.sample((num_embs, token_emb_dim))

        if token_emb_init == "glove":
            token_embs = update_embs_with_glove(tokenizer, token_embs)

        self.token_embedding = nn.Embedding.from_pretrained(
            token_embs,
            freeze=False
        )

        self.var_emb_dim = var_emb_dim
        var_embs = torch.eye(var_emb_dim, dtype=token_embs.dtype)

        self.var_embedding = nn.Embedding.from_pretrained(
            var_embs,
            freeze=True
        )

        self.lstm = nn.LSTM(
            token_emb_dim + var_emb_dim,
            hidden_size,
            num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

    def forward(self, token_idx_seqs, var_idx_seqs, token_var_idx_lens, non_var_idx):
        token_embs = self.token_embedding(token_idx_seqs)

        var_idx_seqs_o  = var_idx_seqs - 1
        in_range_mask   = torch.logical_and(var_idx_seqs_o >= 0, var_idx_seqs_o < self.var_emb_dim)
        var_idx_seqs_om = var_idx_seqs_o * in_range_mask # the rest are zero'd out
        non_var_mask    = (var_idx_seqs == non_var_idx).unsqueeze(dim=-1)
        var_embs = self.var_embedding(var_idx_seqs_om) * non_var_mask

        embs = torch.cat((token_embs, var_embs), dim=-1)

        outputs, hidden_cell = self.lstm(embs)
        hidden, cell = hidden_cell

        #   permute from (num layers * num dirs, batch size, hidden size)
        #             to (batch size, num layers * num dirs, hidden size)
        hidden = hidden.permute(1, 0, 2)
        cell   = cell.permute(1, 0, 2)

        token_var_idx_lens_o   = token_var_idx_lens - 1
        token_var_idx_lens_or  = token_var_idx_lens_o.reshape(-1, 1, 1)
        token_var_idx_lens_orr = token_var_idx_lens_or.repeat(1, 1, outputs.shape[2])

        #   outputs: (batch size, seq len, hidden size)
        outs = torch.gather(outputs, 1, token_var_idx_lens_orr)
        return outs.squeeze(dim=1)

if __name__ == "__main__":
    tokenizer = LFTokenizer()

    syminfo = torch.load("../syminfo.pt")
    tokenizer.load_syminfo(syminfo)
    print(tokenizer.summary())

    lf_model = LFModel(tokenizer, "glove")