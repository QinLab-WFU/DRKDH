import torch


def get_all_triplets_indices(labels):
    sames = labels @ labels.T>0
    diffs = ~sames
    sames.fill_diagonal_(False)
    anc_index,pos_index,neg_indexs = torch.where(sames.unsqueeze(2)*diffs.unsqueeze(1))
    return anc_index,pos_index,neg_indexs