import json
import os
import time

import torch
import torch.nn.functional as F
from loguru import logger


from _data import build_loaders, get_topk, get_class_num
from _network import build_model
from _utils import (
    AverageMeter,
    build_optimizer,
    calc_learnable_params,
    calc_map_eval,
    EarlyStopping,
    init,
    print_in_md,
    save_checkpoint,
    seed_everything,
    validate_smart,
    rename_output,
)
from config import get_config
from loss import (
    RKDLoss,
    RelaxedFacenetLoss,
    RelaxedTripletMarginLoss,
    RelaxedTripletMarginLossMod,
    SoftTripletMarginRegressionLoss,
)


def train_epoch(args, dataloader, cache, net, criterion, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["loss", "main_loss", "quant_loss", "mAP"]:
        stat_meters[x] = AverageMeter()

    net.train()
    for images, labels, idxes in dataloader:
        embeddings = net(images.to(args.device))

        # TODO: normalize labels or nor?
        inputs = F.normalize(labels) if cache is None else cache[idxes]
        # inputs = labels if cache is None else cache[idxes]
        inputs = inputs.to(args.device)

        # just4test!
        # anc_idxes, pos_idxes, neg_idxes = get_all_triplets_indices(labels)
        # triplets = torch.stack([anc_idxes, pos_idxes, neg_idxes], dim=1).view(-1)[:3333]
        # fct = calc_FCT(cache[idxes[triplets]])
        # print(f"FCT: {fct:.3f}")

        # .unsqueeze(1) for multiple teachers

        main_loss = criterion(inputs.unsqueeze(1), embeddings)


        binary_code = embeddings.sign()

        quant_loss = torch.mean((embeddings - binary_code.detach()) ** 2)


        loss = main_loss + args.quant_weight * quant_loss

        stat_meters["main_loss"].update(main_loss)
        stat_meters["quant_loss"].update(quant_loss)
        stat_meters["loss"].update(loss)
        # stat_meters["n_triplets"].update(n_triplets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to check overfitting
        map_v = calc_map_eval(embeddings.sign(), labels.to(args.device))
        stat_meters["mAP"].update(map_v)

        torch.cuda.empty_cache()

    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{sm_str}"
    )


def train_init(args):
    # setup net
    net = build_model(args, True)

    # setup criterion
    if args.loss_type == "rkd":
        criterion = RKDLoss()
    elif args.loss_type == "stmr":
        criterion = SoftTripletMarginRegressionLoss()
    elif args.loss_type == "rtm":
        criterion = RelaxedTripletMarginLoss()
    elif args.loss_type == "rtml":
        criterion = RelaxedTripletMarginLossMod()
    elif args.loss_type == "rf":
        criterion = RelaxedFacenetLoss()
    else:
        raise NotImplementedError

    logger.info(f"Number of learnable params: {calc_learnable_params(net)}")

    # setup optimizer
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)

    return net, criterion, optimizer


def train(args, train_loader, query_loader, dbase_loader, cache):
    net, criterion, optimizer = train_init(args)

    early_stopping = EarlyStopping()

    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, cache, net, criterion, optimizer, epoch)

        # we monitor mAP@topk validation accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                multi_thread=args.multi_thread,
            )
            if early_stop:
                break

    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"Without improvement, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"Reach epoch limit, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )

    save_checkpoint(args, early_stopping.best_checkpoint)

    return early_stopping.best_epoch, early_stopping.best_map


def main():
    init()
    args = get_config()

    if "rename" in args and args.rename:
        rename_output(args)

    dummy_logger_id = None
    rst = []
    for dataset in ["nuswide", "flickr", "imagenet","things"]:
        print(f"Processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = build_loaders(
            dataset, args.data_dir, batch_size=args.batch_size, num_workers=args.n_workers
        )

        cache = None
        if not args.E2E:
            cache_path = f"../DRKDH_STEP1/output/psycho/{dataset}/128/cache.pt"
            cache = torch.load(cache_path)
            print("11")

        for hash_bit in [16, 32, 64, 128]:
            # for hash_bit in [16]:
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

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader, cache)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})
    # for x in rst:
    #     print(
    #         f"[dataset:{x['dataset']}][bits:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
    #     )
    print_in_md(rst)


if __name__ == "__main__":
    main()
