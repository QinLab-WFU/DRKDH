from argparse import Namespace

import torch
from torch import nn

from utils import safe_log, safe_division, calculate_similarities_groups_pt


class InfoNCELoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()

        self.tau = args.tau

    def forward(self, embeddings):
        _, _, dot_products = calculate_similarities_groups_pt(embeddings, 3)

        s_01, s_02 = dot_products

        s_max = torch.maximum(s_01, s_02)
        s_01n = torch.exp((s_01 - s_max) / self.tau)
        s_02n = torch.exp((s_02 - s_max) / self.tau)

        probabilities = safe_division(s_01n, s_01n + s_02n)
        losses = -safe_log(probabilities)

        return torch.mean(losses)


if __name__ == "__main__":
    from _utils import gen_test_data

    B, C, K = 4, 10, 8
    e, t, l = gen_test_data(B, C, K)
