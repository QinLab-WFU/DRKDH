import numpy as np
import torch


class RelaxedSamplingMattersBatchMiner:
    def __init__(self):
        self.tau = 0.4

        self.lower_cutoff = 0.5
        self.upper_cutoff = np.sqrt(2.0)

    def __call__(self, source_embeddings, embeddings):
        n, dim = embeddings.shape
        n2 = n**2

        source_distances_squared = self.pdist(source_embeddings[:, 0, :].detach().cpu(), squared=True)
        source_pair_weights = torch.exp(-source_distances_squared / self.tau)
        source_pair_weights_adjusted = torch.clamp(source_pair_weights - torch.eye(n), min=0.0, max=None)

        distances = self.pdist(embeddings.detach().cpu()).clamp(min=self.lower_cutoff)

        flat_indexes = []
        for i in range(n):
            spwa = source_pair_weights_adjusted[i].numpy()
            # If no positive or no negative, do not use the current anchor.
            if (np.max(spwa) < 0.01) or (np.min(spwa) >= 0.01):
                continue

            # Randomly sample one positive (j).
            p = spwa / np.sum(spwa)  # like sigmoid
            j = np.random.choice(n, p=p)

            # Randomly sample one negative (k) by distance.
            q_d_inv = self.inverse_sphere_distances(dim, distances[i], source_distances_squared[i])
            if np.sum(q_d_inv) == 0.0:
                positives = positives[:-1]
                continue
            p = q_d_inv / np.sum(q_d_inv)
            k = np.random.choice(n, p=p)

            flat_indexes.append(i * n2 + j * n + k)

        return torch.tensor(flat_indexes, device=embeddings.device)

    def inverse_sphere_distances(self, dim, distances, source_distances_squared):
        # negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = (2.0 - dim) * torch.log(distances) + (3.0 - dim) / 2.0 * torch.log(1.0 - 0.25 * distances.pow(2))

        sd2 = torch.clamp(source_distances_squared, min=self.lower_cutoff**2, max=self.upper_cutoff**2)
        offset = 0.5 * (2.0 - dim) * torch.log(sd2) + (3.0 - dim) / 2.0 * torch.log(1.0 - 0.25 * sd2)

        # adjusted for positivity of source labels
        log_q_d_inv -= offset

        q_d_inv = torch.exp(log_q_d_inv - torch.max(log_q_d_inv))

        return q_d_inv.numpy()

    @staticmethod
    def pdist(A, squared=False):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=0)
        if squared:
            return res
        return res.sqrt()
