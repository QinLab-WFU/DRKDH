from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from miner import RelaxedSamplingMattersBatchMiner
from utils import (
    calculate_similarities_pt,
    calculate_source_similarities,
    calculate_triplet_angles_one_hot,
    safe_division,
    safe_log,
    calculate_weights_from_distances,
)


class SoftTripletMarginRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.tau = 5.0
        self.beta = 0.5

        self.batch_miner = RelaxedSamplingMattersBatchMiner()
        self.zero = None

    def forward(self, source_embeddings, embeddings):
        # Calculate distances according to the averaged source and learned embeddings.
        _, source_avg_distances, _, _ = calculate_source_similarities(source_embeddings.detach())
        distances, _, _ = calculate_similarities_pt(embeddings)

        # Calculate triplet losses according to the averaged source and learned embeddings.
        source_triplet_distance_differences = torch.unsqueeze(source_avg_distances, 1) - torch.unsqueeze(
            source_avg_distances, 2
        )
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)

        alpha = torch.sigmoid(source_triplet_distance_differences / self.beta)
        gamma = safe_log(1.0 / alpha - 1.0) / self.tau
        triplet_losses = (
            alpha
            * safe_log(
                1.0 + torch.exp(self.tau * (source_triplet_distance_differences - triplet_distance_differences + gamma))
            )
            + (1.0 - alpha)
            * safe_log(
                1.0 + torch.exp(self.tau * (triplet_distance_differences - source_triplet_distance_differences - gamma))
            )
        ) / self.tau

        flat_indexes = self.batch_miner(source_embeddings, embeddings)
        if len(flat_indexes) > 0:
            loss = torch.mean(torch.take(triplet_losses, flat_indexes))
        else:
            if self.zero is None:
                self.zero = torch.tensor(0.0).to(torch.float).to(embeddings.device)
            loss = self.zero

        return loss


class RKDLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.huber_loss = nn.HuberLoss(reduction="none")
        self.batch_miner = RelaxedSamplingMattersBatchMiner()
        self.zero = None

    def forward(self, source_embeddings, embeddings):
        rkdd_loss = self._compute_rkdd_loss(source_embeddings, embeddings)
        rkda_loss = self._compute_rkda_loss(source_embeddings, embeddings)
        return 1.0 * rkdd_loss + 2.0 * rkda_loss

    def _compute_rkdd_loss(self, source_embeddings, embeddings):
        # Calculate distances according to the averaged source and learned embeddings.
        _, source_avg_distances, _, _ = calculate_source_similarities(source_embeddings.detach())
        distances, _, _ = calculate_similarities_pt(embeddings)

        # Calculate mean distances according to the averaged source and learned embeddings.
        n = source_embeddings.shape[0]
        source_mean_avg_distances = torch.sum(source_avg_distances, dim=1, keepdims=True) / (n - 1)
        mean_distances = torch.sum(distances, dim=1, keepdims=True) / (n - 1)

        # Calculate the Huber loss for the distances normalized by the mean.
        losses = self.huber_loss(
            safe_division(distances, mean_distances),
            safe_division(source_avg_distances, source_mean_avg_distances),
        )
        loss = torch.mean(losses)

        return loss

    def _compute_rkda_loss(self, source_embeddings, embeddings):
        source_angles = []
        for i in range(source_embeddings.shape[1]):
            se = source_embeddings[:, i, :].detach()
            a = calculate_triplet_angles_one_hot(se)
            source_angles.append(a)
        source_angles = torch.mean(torch.stack(source_angles), dim=0)

        angles = calculate_triplet_angles_one_hot(embeddings)

        losses = self.huber_loss(angles, source_angles)
        flat_indexes = self.batch_miner(source_embeddings, embeddings)
        if len(flat_indexes) > 0:
            loss = torch.mean(torch.take(losses, flat_indexes))
        else:
            if self.zero is None:
                self.zero = torch.tensor(0.0).to(torch.float).to(embeddings.device)
            loss = self.zero

        return loss


class RelaxedTripletMarginLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.margin = 0.5
        self.tau = 0.25

    def forward(self, source_embeddings, embeddings):
        """
        see Eq. 5
        """
        # Calculate distances according to the source embeddings.
        _, source_avg_distances, _, _ = calculate_source_similarities(source_embeddings.detach())

        # Assign weights to triplets according to the source embeddings.
        _, _, source_triplet_weights = calculate_weights_from_distances(source_avg_distances, self.tau)

        # Calculate distances according to the embeddings.
        distances, _, _ = calculate_similarities_pt(embeddings)

        # Calculate triplet losses based on distances: d_an - d_ap
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)
        # [d_ap - d_an + m]+
        triplet_losses = F.relu(self.margin - triplet_distance_differences)

        # Calculate the loss based on the triplet losses weighted by source embedding derived weights.
        loss = safe_division(torch.sum(triplet_losses * source_triplet_weights), torch.sum(source_triplet_weights))

        return loss


class RelaxedTripletMarginLossMod(nn.Module):
    def __init__(self):
        super().__init__()

        self.margin = 0.5
        self.tau = 0.25  # -> STMR.beta

    def forward(self, source_embeddings, embeddings):
        """
        see Eq. 5
        """
        # Calculate distances according to the source embeddings.
        _, source_avg_distances, _, _ = calculate_source_similarities(source_embeddings.detach())

        # Assign weights to triplets according to the source embeddings.
        _, _, source_triplet_weights = calculate_weights_from_distances(source_avg_distances, self.tau)

        # Calculate distances according to the embeddings.
        distances, _, _ = calculate_similarities_pt(embeddings)

        # Calculate triplet losses based on distances: d_an - d_ap
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)
        # [d_ap - d_an + m]+
        triplet_losses = F.relu(self.margin - triplet_distance_differences)

        # mod begin...
        mask = (triplet_losses > 0).float()  # only has grad
        # mask *= 1 - torch.eye(mask.shape[0], device=mask.device).unsqueeze(-1)  # exclude self-cmp
        source_triplet_weights *= mask
        # mod end...

        # Calculate the loss based on the triplet losses weighted by source embedding derived weights.
        loss = safe_division(torch.sum(triplet_losses * source_triplet_weights), torch.sum(source_triplet_weights))

        return loss


class RelaxedFacenetLoss(nn.Module):
    RHO = 10.0

    def __init__(self):
        super().__init__()

        self.margin = 0.5
        self.tau = 0.25
        self.semihard_loss_threshold = 0.0

    def forward(self, source_embeddings, embeddings):
        # Calculate distances according to the source embeddings.
        _, source_avg_distances, _, _ = calculate_source_similarities(source_embeddings.detach())

        # Assign weights to pairs and triplets according to the source embeddings.
        source_pair_weights, _, source_triplet_weights = calculate_weights_from_distances(
            source_avg_distances, self.tau
        )

        # Calculate distances according to the embeddings.
        distances, _, _ = calculate_similarities_pt(embeddings)

        # Calculate triplet losses according to the embeddings: d_an - d_ap
        triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)
        # [d_ap - d_an + m]+
        triplet_losses = F.relu(self.margin - triplet_distance_differences)

        # Calculate positive pair semihard losses based on distances.
        semihard_triplet = (
            torch.logical_and(triplet_losses > 0.0, triplet_losses <= self.margin) * source_triplet_weights
        )
        positive_pair_semihard_losses = torch.amax(triplet_losses * semihard_triplet, dim=2)

        # If required, calculate positive pair hard losses based on distances.
        if self.semihard_loss_threshold > 0.0:
            hard_triplet = (triplet_losses > self.margin) * source_triplet_weights
            normalized_losses = torch.amin(triplet_losses - self.RHO * hard_triplet, dim=2)
            positive_pair_hard_losses = (normalized_losses + self.RHO) * (normalized_losses < 0.0)
        else:
            positive_pair_hard_losses = 0.0

        # Calculate final loss based on the positive pair losses weighted by source embedding derived weights.
        positive_pair_losses = (
            positive_pair_semihard_losses
            + (positive_pair_semihard_losses <= self.semihard_loss_threshold) * positive_pair_hard_losses
        )
        loss = safe_division(torch.sum(positive_pair_losses * source_pair_weights), torch.sum(source_pair_weights))

        return loss


if __name__ == "__main__":
    from _utils import gen_test_data

    B, C, K = 5, 10, 8
    e, _, l = gen_test_data(B, C, K, is_multi_hot=False)
    l = l.unsqueeze(1)

    criterion = RelaxedTripletMarginLossMod()
    loss = criterion(l, e)
    print(loss)
