from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn


def build_model(args: Namespace, pretrained=True):
    if args.backbone == "psycho":
        net = Psycho(args.n_samples, args.n_bits, args.beta, pretrained).to(args.device)
        return net

    raise NotImplementedError(f"Not support: {args.backbone}")


class Psycho(nn.Module):
    def __init__(self, n_instances, n_bits, beta, pretrained):
        super().__init__()

        if pretrained:
            initial_weights = torch.normal(mean=0.5, std=0.2, size=(n_instances, n_bits))
            self.embedding_layer = nn.Embedding.from_pretrained(initial_weights, freeze=False)
        else:
            self.embedding_layer = nn.Embedding(n_instances, n_bits)
        self.beta = beta

    def forward(self, x):
        x = self.embedding_layer(x)
        x = F.relu(x)

        x = F.normalize(x, dim=-1)

        return x

    def custom_regularized_parameters(self):
        return self.embedding_layer.parameters()

    def get_custom_regularization_loss(self):
        regularization_losses = [self.beta * torch.abs(p.data) for p in self.embedding_layer.parameters()]

        return torch.sum(torch.stack(regularization_losses))


if __name__ == "__main__":
    B, N, K = 4, 100, 8
    args = Namespace(backbone="psycho", n_samples=N, n_bits=K, beta=0.1, device="cuda:1")
    net = build_model(args)
    x = torch.randint(0, N, (B,))
    y = net(x.to(args.device))
    print(y.shape)
