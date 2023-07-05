import torch
import torch.nn as nn
from torch.utils.data import DistributedSampler, DataLoader
import torch.distributed as dist
import argparse
import accelerate
from kn_util.basic import registry, global_get, global_set
from kn_util.utils import match_name_keywords, lazyconf2str
from kn_util.config import LazyConfig, instantiate
import kn_util.distributed as kn_dist
from loguru import logger
from trainer import TrainerSimple, TrainerNativeDDP
import wandb
from omegaconf import OmegaConf, DictConfig
from beartype import beartype
from evaluate import evaluate
import os.path as osp
from typing import Dict
import sys
from kn_util.debug import register_hooks_recursively, check_forward_register
import os
from accelerate import DistributedDataParallelKwargs


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--cfg", type=str, required=True)
    args.add_argument("--opts", type=str, nargs="+", default=[])
    args.add_argument("--debug", action="store_true", default=False)
    args.add_argument("--exp", type=str, required=True)
    args.add_argument("--resume", type=str, default=None)
    args.add_argument("--amp", action="store_true", default=False)
    args.add_argument("--wandb", action="store_true", default=False)
    args.add_argument("--overfit", action="store_true", default=False)
    args.add_argument("--eval_first", action="store_true", default=False)
    args.add_argument("--find_unused_parameters", action="store_true", default=False)

    return args.parse_args()


@beartype
def wandb_init(project="contrastive_detr", cfg: DictConfig = None, mode="online"):
    wandb.init(project=project, config=OmegaConf.to_container(cfg, resolve=True), name=cfg.args.exp, mode=mode)
    wandb.run.log_code(cfg.paths.root,
                       include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml") or path.endswith(".yml"))


def get_lr_schema(model, lr_schema: Dict[str, float] = None, base_lr=1e-4):
    if lr_schema is None:
        return model.parameters()
    param_schema = []
    for keyword in lr_schema:
        coeff = lr_schema[keyword]
        cur_dict = dict(params=[param for name, param in model.named_parameters() if keyword in name],
                        lr=coeff * base_lr)
        param_schema.append(cur_dict)

    not_match_any = lambda name: not any([keyword in name for keyword in lr_schema])

    params = [param for name, param in model.named_parameters() if not_match_any(name)]
    if len(params) > 0:
        param_schema.append(dict(params=params, lr=base_lr))

    return param_schema


def setup_cfg(cfg, args):
    LazyConfig.apply_overrides(cfg, args.opts)
    cfg.args = vars(args)

    if args.debug:
        cfg.runtime.num_workers = 0
        cfg.runtime.prefetch_factor = None


def setup_logger(logger, work_dir):
    # Remove default logger, create a logger for each process which writes to a
    # separate log-file. This makes changes in global scope.
    logger.remove(0)
    if kn_dist.get_world_size() > 1:
        logger.add(
            os.path.join(work_dir, f"log-rank{kn_dist.get_rank()}.log"),
            format="{time} {level} {message}",
        )

    # Add a logger for stdout only for the master process.
    if kn_dist.is_master_process():
        logger.add(sys.stdout, format="<g>{time}</g>: <lvl>{message}</lvl>", colorize=True)

    logger.info(f"Current process: Rank {kn_dist.get_rank()}, World size {kn_dist.get_world_size()}")


def main():
    args = parse_args()
    cfg = LazyConfig.load(args.cfg)
    setup_cfg(cfg, args)
    os.makedirs(cfg.paths.work_dir, exist_ok=True)
    setup_logger(logger, work_dir=cfg.paths.work_dir)
    wandb_init(cfg=cfg, mode="disabled" if not args.wandb else "online")

    logger.info(f"Args\n:{lazyconf2str(cfg.args)}")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision="fp16" if args.amp else None,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=cfg.runtime.find_unused_parameters)])

    # upload to global
    global_set("accelerator", accelerator)

    train_loader = instantiate(cfg.train_loader)
    eval_loader = instantiate(cfg.eval_loader)
    test_loader = instantiate(cfg.test_loader)
    model = instantiate(cfg.model)

    param_schema = get_lr_schema(model, cfg.get("lr_schema", None), base_lr=cfg.optimizer.lr)
    optimizer = instantiate(cfg.optimizer, params=param_schema, _convert_="partial")
    lr_scheduler = None if "lr_scheduler" not in cfg else instantiate(cfg.lr_scheduler, optimizer=optimizer)
    # checkpointer = instantiate(cfg.checkpointer)

    train_evaluater = instantiate(cfg.train_evaluater)
    eval_evaluater = instantiate(cfg.eval_evaluater)
    test_evaluater = instantiate(cfg.test_evaluater)

    model, train_loader, eval_loader, test_loader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_loader, eval_loader, test_loader, optimizer, lr_scheduler)

    # register_hooks_recursively(model, hook_registers=[check_forward_register])

    if args.overfit:
        cfg.train.train_print_interval = 1
        cfg.train.eval_interval = 1
        trainer = TrainerSimple(optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                lr_schedule_after=cfg.train.get("lr_schedule_after", None),
                                lr_schedule_monitor=cfg.train.get("lr_schedule_monitor", None),
                                train_evaluater=train_evaluater,
                                eval_evaluater=eval_evaluater)
        global_set("trainer", trainer)
        trainer.overfit(model, train_loader, num_epochs=1000, sample_batch=1)
    else:
        trainer = TrainerSimple(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_schedule_after=cfg.train.get("lr_schedule_after", None),
            lr_schedule_monitor=cfg.train.get("lr_schedule_monitor", None),
            train_evaluater=train_evaluater,
            eval_evaluater=eval_evaluater,
            interval_eval=cfg.train.interval_eval,
            interval_log_train=cfg.train.interval_log_train,
        )
        global_set("trainer", trainer)

        logger.info("Sanity check on evaluation")
        trainer.eval_sanity_check(model, eval_loader, sample_batch=3)
        trainer.train(model, train_loader, eval_loader, num_epochs=cfg.train.num_epochs)


def main_native():
    dist.init_process_group(backend="nccl", init_method="env://", world_size=kn_dist.get_world_size())
    device = torch.device(f"cuda:{kn_dist.get_rank()}")
    torch.cuda.set_device(device)
    global_set("device", device)

    def convert_dataloader_distributed(dl, shuffle=True):
        dataset = dl.dataset
        batch_size = dl.batch_size
        num_workers = dl.num_workers
        pin_memory = dl.pin_memory
        drop_last = dl.drop_last
        collate_fn = dl.collate_fn
        sampler = DistributedSampler(dataset,
                                     num_replicas=kn_dist.get_world_size(),
                                     rank=kn_dist.get_rank(),
                                     shuffle=shuffle,
                                     drop_last=drop_last)
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          sampler=sampler,
                          collate_fn=collate_fn)

    args = parse_args()
    cfg = LazyConfig.load(args.cfg)
    setup_cfg(cfg, args)
    os.makedirs(cfg.paths.work_dir, exist_ok=True)
    setup_logger(logger, work_dir=cfg.paths.work_dir)
    wandb_init(cfg=cfg, mode="disabled" if not args.wandb else "online")

    train_loader = instantiate(cfg.train_loader)
    eval_loader = instantiate(cfg.eval_loader)
    test_loader = instantiate(cfg.test_loader)
    model = instantiate(cfg.model).cuda()

    param_schema = get_lr_schema(model, cfg.get("lr_schema", None), base_lr=cfg.optimizer.lr)
    optimizer = instantiate(cfg.optimizer, params=param_schema, _convert_="partial")
    lr_scheduler = None if "lr_scheduler" not in cfg else instantiate(cfg.lr_scheduler, optimizer=optimizer)
    # checkpointer = instantiate(cfg.checkpointer)

    train_evaluater = instantiate(cfg.train_evaluater)
    eval_evaluater = instantiate(cfg.eval_evaluater)
    test_evaluater = instantiate(cfg.test_evaluater)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[kn_dist.get_rank()])
    train_loader, eval_loader = [convert_dataloader_distributed(dl) for dl in [train_loader, eval_loader]]

    trainer = TrainerNativeDDP(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_schedule_after=cfg.train.get("lr_schedule_after", None),
        lr_schedule_monitor=cfg.train.get("lr_schedule_monitor", None),
        train_evaluater=train_evaluater,
        eval_evaluater=eval_evaluater,
        interval_eval=cfg.train.interval_eval,
        interval_log_train=cfg.train.interval_log_train,
    )
    global_set("trainer", trainer)

    logger.info("Sanity check on evaluation")
    trainer.eval_sanity_check(model, eval_loader, sample_batch=3)
    trainer.train(model, train_loader, eval_loader, num_epochs=cfg.train.num_epochs)


if __name__ == "__main__":
    main()
    # main_native()