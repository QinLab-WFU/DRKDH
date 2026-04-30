import torch

EPSILON, YPSILON = 1e-12, 1e12


def safe_division(x, positive_y):
    return x / (positive_y + EPSILON)


def safe_sqrt(quasi_positive_x):
    return torch.sqrt(torch.clamp(quasi_positive_x, min=EPSILON))


def safe_log(x):
    return torch.log(torch.clamp(x, min=EPSILON, max=YPSILON))


def calculate_source_similarities(source_embeddings):
    source_distances, source_dot_products = [], []

    for i in range(source_embeddings.shape[1]):
        d, _, dp = calculate_similarities_pt(source_embeddings[:, i, :])
        source_distances.append(d)
        source_dot_products.append(dp)

    source_avg_distances = torch.mean(torch.stack(source_distances), dim=0)
    source_avg_dot_products = torch.mean(torch.stack(source_dot_products), dim=0)

    return source_distances, source_avg_distances, source_dot_products, source_avg_dot_products


def calculate_similarities_pt(embeddings):
    """
    It actually calculates distance between two embeddings.
    """
    dot_products = torch.mm(embeddings, embeddings.t())
    squared_norms = dot_products.diag().unsqueeze(1).expand_as(dot_products)
    # (2 - 2 * embeddings @ embeddings.T).clamp(min=0.0) # if embeddings is normed
    squared_distances = (squared_norms + squared_norms.t() - 2.0 * dot_products).clamp(min=0.0)
    distances = safe_sqrt(squared_distances)

    return distances, squared_distances, dot_products


def calculate_triplet_angles_one_hot(embeddings):
    classes = torch.argmax(embeddings, 1)
    same_class = (torch.unsqueeze(classes, 0) == torch.unsqueeze(classes, 1)).to(torch.float)
    angles = 0.5 * (
        1.0 + torch.unsqueeze(same_class, 0) - torch.unsqueeze(same_class, 2) - torch.unsqueeze(same_class, 1)
    )

    return angles


def calculate_weights_from_distances(distances, tau, margin=0.0):
    """
    Note: this margin is not the margin in TripletLoss!
    TODO: distances contains self-compare
    """
    pair_weights = torch.exp(-distances / tau)

    pair_weights_negated = 1.0 - pair_weights
    pair_weights.fill_diagonal_(0.0)
    pair_weights_negated.fill_diagonal_(0.0)

    triplet_distance_differences = torch.unsqueeze(distances, 1) - torch.unsqueeze(distances, 2)
    # sigmoid((d_an - d_ap) / tau) -> triplet or not
    triplet_weights = torch.sigmoid((triplet_distance_differences - margin) / tau)

    return pair_weights, pair_weights_negated, triplet_weights
