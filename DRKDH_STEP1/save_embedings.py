import glob

import torch
from loguru import logger
from tqdm import tqdm

from _data import build_loaders
from _utils import init
from config import get_config
from network import build_model


def predict(net, dataloader, verbose=True):
    device = next(net.parameters()).device

    codes, idxes = [], []
    net.eval()

    if verbose:
        try:
            _iter = tqdm(dataloader, desc=f"extracting {dataloader.dataset.usage} features")
        except AttributeError:
            _iter = tqdm(dataloader, desc=f"extracting features")
    else:
        _iter = dataloader

    for _, _, inds in _iter:
        with torch.no_grad():
            embs = net(inds.to(device))

        codes.append(embs)
        idxes.append(inds)

    return torch.cat(codes), torch.cat(idxes).to(device)


if __name__ == "__main__":
    init()

    args = get_config()
    args.device = "cuda:1"

    rst = []
    for dataset in ["nuswide", "flickr", "imagenet","things"]:
        logger.info(f"Processing dataset: {dataset}")
        args.dataset = dataset

        train_loader, _, _ = build_loaders(dataset, args.data_dir, batch_size=1, num_workers=args.n_workers)
        args.n_samples = train_loader.dataset.__len__()

        for hash_bit in [128]:
            logger.info(f"Processing hash-bit: {hash_bit}")
            args.n_bits = hash_bit

            net = build_model(args, False)

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            pkl_list = glob.glob(f"{args.save_dir}/*.pth")
            if len(pkl_list) != 1:
                logger.error(pkl_list)
                raise Exception(f"Cannot locate one *.pth in: {args.save_dir}")

            checkpoint = torch.load(pkl_list[0], map_location="cpu")
            msg = net.load_state_dict(checkpoint["model"])
            logger.info(f"Model loaded: {msg}")

            codes, idxes = predict(net, train_loader)
            reverse_idxes = torch.argsort(idxes)

            torch.save(codes[reverse_idxes].cpu(), f"{args.save_dir}/cache.pt")

    print("done!")
