from kn_util.basic import global_get, global_set
from kn_util.utils import max_memory_allocated, dict2str, module2tree
import kn_util.distributed as dist
from kn_util.utils import CheckPointer
import kn_util.distributed as kn_dist
from loguru import logger
from evaluate import evaluate, evaluate_native
import time
import torch
import torch.nn as nn
from loguru import logger
import wandb
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


def must_provided_when(when, **kwargs):
    if when:
        assert any((kwargs.get(k) is not None for k in kwargs))


def call_when(when, func, *args, **kwargs):
    if when:
        return func(*args, **kwargs)
    else:
        return None


def default(val, default_val):
    return val if val is not None else default_val


class TrainerSimple:

    def __init__(self,
                 optimizer=None,
                 lr_scheduler=None,
                 train_evaluater=None,
                 interval_log_train=0.1,
                 interval_eval=1.0,
                 lr_schedule_after=None,
                 lr_schedule_monitor=None,
                 eval_evaluater=None,
                 test_evaluater=None,
                 do_train=True,
                 do_eval=True,
                 eval_only=False,
                 clip_grad_norm=10,
                 do_test=False,
                 start_epoch=0):
        """
        Args:
            lr_schedule_by: "epoch" or "step" or "eval"
        """
        self.accelerator = global_get("accelerator")
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_schedule_after = lr_schedule_after
        self.lr_schedule_moniter = lr_schedule_monitor
        self.train_evaluater = train_evaluater
        self.eval_evaluater = eval_evaluater
        self.test_evaluater = test_evaluater
        self.interval_log_train = interval_log_train
        self.interval_eval = interval_eval
        self.clip_grad_norm = clip_grad_norm
        self.do_train, self.do_eval, self.do_test = do_train, do_eval, do_test

        self.epoch = start_epoch
        self.global_step = 0

        must_provided_when(do_train, optimizer=optimizer, train_evaluater=train_evaluater)
        must_provided_when(do_eval or eval_only, eval_evaluater=eval_evaluater)
        must_provided_when(lr_schedule_after is not None, lr_scheduler=lr_scheduler)
        must_provided_when(lr_scheduler is not None and isinstance(lr_scheduler, ReduceLROnPlateau),
                           lr_schedule_moniter=lr_schedule_monitor)

    def evaluate(self, model, data_loader, evaluater=None):
        evaluater = default(evaluater, self.eval_evaluater)

        unwrap_model = self.accelerator.unwrap_model(model)
        metrics = evaluate(unwrap_model, data_loader, evaluater)
        return metrics

    def evaluate_and_log(self, model, data_loader, evaluater=None):
        metrics = self.evaluate(model, data_loader, evaluater)
        wandb.log(metrics, step=self.global_step)
        if self.do_train:
            logger.info(f"Evaluate Epoch {self.epoch} [{self.batch_idx}/{self.num_batch}]]")
        logger.info(dict2str(metrics, delim=": ", sep="\t", exclude_keys=["lr"], fmt=".4g"))
        return metrics

    def log_train(self, train_evaluater, optimizer):
        train_print_dict = train_evaluater.compute_all()
        train_print_dict["lr"] = optimizer.param_groups[0]["lr"]
        train_print_dict["time"] = time.time() - self.st
        train_print_dict["mem"] = max_memory_allocated()
        wandb.log(train_print_dict, step=self.global_step)
        logger.info(f"Epoch {self.epoch} [{self.batch_idx}/{self.num_batch}]")
        logger.info(dict2str(train_print_dict, delim=": ", sep="\t", fmt=".4g"))

    def train_one_epoch(self, model, train_loader, eval_loader):
        self.st = time.time()
        train_evaluater = self.train_evaluater
        optimizer = self.optimizer
        eval_evaluater = self.eval_evaluater
        accelerator = self.accelerator

        self.num_batch = num_batch = len(train_loader)
        step_interval_log_train = int(num_batch * self.interval_log_train)
        step_interval_eval = int(num_batch * self.interval_eval)

        must_provided_when(self.do_train, train_loader=train_loader)
        must_provided_when(self.do_eval, eval_loader=eval_loader)

        for batch_idx, batch in enumerate(train_loader):
            self.batch_idx = batch_idx
            model.train()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            global_set("batch", batch)  #! convenient for debugging

            loss_dict = model(**batch)
            loss = loss_dict["loss"]
            train_evaluater.update_all(loss_dict)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if (self.global_step + 1) % step_interval_log_train == 0:
                self.log_train(train_evaluater, optimizer)

            if (self.global_step + 1) % step_interval_eval == 0:
                metrics = self.evaluate_and_log(model, eval_loader, eval_evaluater)
                if self.lr_schedule_after == "eval":
                    self.lr_scheduler.step(metrics[self.lr_schedule_moniter])

            self.global_step += 1

    def log_info(self):
        logger.info(f"============== Model Summary ==================\n{module2tree(self.model)}")
        logger.info(f"============== Optim Summary ==================\n{self.optimizer}\n{self.lr_scheduler}")
        logger.info(f"======= Start Training for {self.num_epochs} epochs =======")

    def eval_sanity_check(self, model, eval_loader, sample_batch):
        eval_loader = [next(iter(eval_loader)) for _ in range(sample_batch)]
        metrics = self.evaluate(model, eval_loader, self.eval_evaluater)
        logger.info(dict2str(metrics, delim=": ", sep="\t", fmt=".4g"))

    def train(self, model, train_loader, eval_loader=None, num_epochs=1):
        self.model = model
        self.num_epochs = num_epochs
        self.log_info()

        for epoch in range(num_epochs):
            self.train_one_epoch(
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader,
            )
            self.epoch += 1

    def overfit(self, model, train_loader, sample_batch, num_epochs=1):
        self.model = model
        self.num_epochs = num_epochs
        self.log_info()

        self.interval_log_train = 1.0
        self.interval_eval = 1.0

        self.global_step = 0
        train_loader = [next(iter(train_loader)) for _ in range(sample_batch)]
        for epoch in range(num_epochs):
            self.train_one_epoch(model=model, train_loader=train_loader, eval_loader=train_loader)
            self.epoch += 1


class TrainerNativeDDP:

    def __init__(self,
                 optimizer=None,
                 lr_scheduler=None,
                 train_evaluater=None,
                 interval_log_train=0.1,
                 interval_eval=1.0,
                 lr_schedule_after=None,
                 lr_schedule_monitor=None,
                 eval_evaluater=None,
                 test_evaluater=None,
                 do_train=True,
                 do_eval=True,
                 eval_only=False,
                 clip_grad_norm=10,
                 do_test=False,
                 start_epoch=0):
        """
        Args:
            lr_schedule_by: "epoch" or "step" or "eval"
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_schedule_after = lr_schedule_after
        self.lr_schedule_moniter = lr_schedule_monitor
        self.train_evaluater = train_evaluater
        self.eval_evaluater = eval_evaluater
        self.test_evaluater = test_evaluater
        self.interval_log_train = interval_log_train
        self.interval_eval = interval_eval
        self.clip_grad_norm = clip_grad_norm
        self.do_train, self.do_eval, self.do_test = do_train, do_eval, do_test

        self.epoch = start_epoch
        self.global_step = 0

        must_provided_when(do_train, optimizer=optimizer, train_evaluater=train_evaluater)
        must_provided_when(do_eval or eval_only, eval_evaluater=eval_evaluater)
        must_provided_when(lr_schedule_after is not None, lr_scheduler=lr_scheduler)
        must_provided_when(lr_scheduler is not None and isinstance(lr_scheduler, ReduceLROnPlateau),
                           lr_schedule_moniter=lr_schedule_monitor)

    def evaluate(self, model, data_loader, evaluater=None):
        evaluater = default(evaluater, self.eval_evaluater)

        unwrap_model = model.module
        metrics = evaluate_native(unwrap_model, data_loader, evaluater)
        return metrics

    def evaluate_and_log(self, model, data_loader, evaluater=None):
        metrics = self.evaluate(model, data_loader, evaluater)
        wandb.log(metrics, step=self.global_step)
        if self.do_train:
            logger.info(f"Evaluate Epoch {self.epoch} [{self.batch_idx}/{self.num_batch}]]")
        logger.info(dict2str(metrics, delim=": ", sep="\t", exclude_keys=["lr"], fmt=".4g"))
        return metrics

    def log_train(self, train_evaluater, optimizer):
        train_print_dict = train_evaluater.compute_all()
        train_print_dict["lr"] = optimizer.param_groups[0]["lr"]
        train_print_dict["time"] = time.time() - self.st
        train_print_dict["mem"] = max_memory_allocated()
        wandb.log(train_print_dict, step=self.global_step)
        logger.info(f"Epoch {self.epoch} [{self.batch_idx}/{self.num_batch}]")
        logger.info(dict2str(train_print_dict, delim=": ", sep="\t", fmt=".4g"))

    def train_one_epoch(self, model, train_loader, eval_loader):
        self.st = time.time()
        train_evaluater = self.train_evaluater
        optimizer = self.optimizer
        eval_evaluater = self.eval_evaluater
        device = global_get("device")

        self.num_batch = num_batch = len(train_loader)
        step_interval_log_train = int(num_batch * self.interval_log_train)
        step_interval_eval = int(num_batch * self.interval_eval)

        must_provided_when(self.do_train, train_loader=train_loader)
        must_provided_when(self.do_eval, eval_loader=eval_loader)

        for batch_idx, batch in enumerate(train_loader):
            self.batch_idx = batch_idx
            model.train()
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            global_set("batch", batch)  #! convenient for debugging

            loss_dict = model(**batch)
            loss = loss_dict["loss"]
            train_evaluater.update_all(loss_dict)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if (self.global_step + 1) % step_interval_log_train == 0:
                self.log_train(train_evaluater, optimizer)

            if (self.global_step + 1) % step_interval_eval == 0:
                metrics = self.evaluate_and_log(model, eval_loader, eval_evaluater)
                if self.lr_schedule_after == "eval":
                    self.lr_scheduler.step(metrics[self.lr_schedule_moniter])

            self.global_step += 1

    def log_info(self):
        logger.info(f"============== Model Summary ==================\n{self.model}")
        logger.info(f"============== Optim Summary ==================\n{self.optimizer}\n{self.lr_scheduler}")
        logger.info(f"======= Start Training for {self.num_epochs} epochs =======")

    def eval_sanity_check(self, model, eval_loader, sample_batch):
        eval_loader = [next(iter(eval_loader)) for _ in range(sample_batch)]
        metrics = self.evaluate(model, eval_loader, self.eval_evaluater)
        logger.info(dict2str(metrics, delim=": ", sep="\t", fmt=".4g"))

    def train(self, model, train_loader, eval_loader=None, num_epochs=1):
        self.model = model
        self.num_epochs = num_epochs
        self.log_info()

        for epoch in range(num_epochs):
            self.train_one_epoch(
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader,
            )
            self.epoch += 1

    def overfit(self, model, train_loader, sample_batch, num_epochs=1):
        self.model = model
        self.num_epochs = num_epochs
        self.log_info()

        self.interval_log_train = 1.0
        self.interval_eval = 1.0

        self.global_step = 0
        train_loader = [next(iter(train_loader)) for _ in range(sample_batch)]
        for epoch in range(num_epochs):
            self.train_one_epoch(model=model, train_loader=train_loader, eval_loader=train_loader)
            self.epoch += 1
