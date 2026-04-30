import torch

EPSILON, YPSILON = 1e-12, 1e12


def get_all_triplets_indices(labels):
    sames = labels @ labels.T > 0
    diffs = ~sames
    sames.fill_diagonal_(False)

    anc_idxes, pos_idxes, neg_idxes = torch.where(sames.unsqueeze(2) * diffs.unsqueeze(1))

    return anc_idxes, pos_idxes, neg_idxes


def safe_division(x, positive_y):
    return x / (positive_y + EPSILON)


def safe_sqrt(quasi_positive_x):
    return torch.sqrt(torch.clamp(quasi_positive_x, min=EPSILON))


def safe_log(x):
    return torch.log(torch.clamp(x, min=EPSILON, max=YPSILON))


def calculate_similarities_groups_pt(embeddings, group_size):
    # [anchors, positives, negatives]
    embeddings_by_group_offset = [embeddings[i : embeddings.shape[0] : group_size, :] for i in range(group_size)]

    dot_products = []
    squared_distances = []
    distances = []

    ijs = [(0, j) for j in range(1, group_size)]  # [(0,1)->ap, (0,2)->an]
    for i, j in ijs:
        dot_products_ij = torch.sum(embeddings_by_group_offset[i] * embeddings_by_group_offset[j], dim=1)
        squared_norms_i = torch.sum(embeddings_by_group_offset[i] ** 2, dim=1)
        squared_norms_j = torch.sum(embeddings_by_group_offset[j] ** 2, dim=1)
        squared_distances_ij = (squared_norms_i + squared_norms_j - 2.0 * dot_products_ij).clamp(min=0.0)
        distances_ij = safe_sqrt(squared_distances_ij)

        dot_products.append(dot_products_ij)
        squared_distances.append(squared_distances_ij)
        distances.append(distances_ij)

    return distances, squared_distances, dot_products


def calc_FCT(embeddings):
    """
    fraction of correct triplets
    """
    distances, _, _ = calculate_similarities_groups_pt(embeddings, 3)
    d_01, d_02 = distances  # d_ap, d_an
    correct_triplets = d_02 > d_01  # d_an > d_ap
    return correct_triplets.float().mean()
