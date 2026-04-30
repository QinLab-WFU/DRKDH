import json
import os
import time

import torch
from loguru import logger


from _data import build_loaders, get_topk, get_class_num
from _utils import (
    AverageMeter,
    build_optimizer,
    calc_learnable_params,
    init,
    print_in_md,
    save_checkpoint,
    seed_everything,
)
from config import get_config
from loss import InfoNCELoss
from network import build_model
from utils import calc_FCT, get_all_triplets_indices


def train_epoch(args, dataloader, net, criterion, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["n_triplets", "nce_loss", "reg_loss", "loss", "FCT"]:
        stat_meters[x] = AverageMeter()

    net.train()

    for _, labels, indices in dataloader:
        labels, indices = labels.to(args.device), indices.to(args.device)

        anc_idxes, pos_idxes, neg_idxes = get_all_triplets_indices(labels)
        stat_meters["n_triplets"].update(anc_idxes.numel())
        triplets = indices[torch.stack([anc_idxes, pos_idxes, neg_idxes], dim=1).view(-1)]

        embeddings = net(triplets)

        nce_loss = criterion(embeddings)
        stat_meters["nce_loss"].update(nce_loss)

        reg_loss = net.get_custom_regularization_loss()
        stat_meters["reg_loss"].update(reg_loss)

        loss = nce_loss + reg_loss
        stat_meters["loss"].update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to check overfitting
        fct = calc_FCT(embeddings)
        stat_meters["FCT"].update(fct)

        torch.cuda.empty_cache()

    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{sm_str}"
    )

    return stat_meters["FCT"].avg


def train_init(args):
    # setup net
    net = build_model(args, True)

    # setup criterion
    criterion = InfoNCELoss(args)

    logger.info(f"Number of learnable params: {calc_learnable_params(net)}")

    # setup optimizer
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)

    return net, criterion, optimizer


def train(args, train_loader, _, __):
    net, criterion, optimizer = train_init(args)

    final_fct = 0
    for epoch in range(args.n_epochs):
        final_fct = train_epoch(args, train_loader, net, criterion, optimizer, epoch)

    logger.info(f"reach epoch limit, will save & exit")

    save_checkpoint(args, {"model": net.state_dict(), "epoch": args.n_epochs - 1, "map": final_fct})

    return args.n_epochs - 1, final_fct


def main():
    init()
    args = get_config()

    # rename_output(args)

    dummy_logger_id = None
    rst = []
    for dataset in ["nuswide", "flickr", "imagenet","things"]:
        print(f"Processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = build_loaders(
            args.dataset, args.data_dir, batch_size=args.batch_size, num_workers=args.n_workers
        )
        args.n_samples = train_loader.dataset.__len__()

        # for hash_bit in [16, 32, 64, 128]:
        for hash_bit in [128]:
            print(f"Processing hash-bit: {hash_bit}")
            seed_everything()
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pth") for x in os.listdir(args.save_dir)):
                print(f"*.pth exists in {args.save_dir}, will pass")
                continue

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")

            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(
                    vars(args),
                    f,
                    indent=4,
                    sort_keys=True,
                    default=lambda o: o if type(o) in [bool, int, float, str] else str(type(o)),
                )

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})

    print_in_md(rst)


if __name__ == "__main__":
    main()
