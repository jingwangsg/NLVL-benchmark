import os.path as osp
import copy
from torch.utils.data import DataLoader
from kn_util.config import LazyCall as L


def build_dataloaders(dataset,
                      dataset_mapper,
                      prefetch_factor="${runtime.prefetch_factor}",
                      num_workers="${runtime.num_workers}",
                      batch_size="${train.batch_size}",
                      val_split="test"):
    dataset["split"] = "train"
    train_loader = L(DataLoader)(dataset=dataset,
                                 prefetch_factor=prefetch_factor,
                                 num_workers=num_workers,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=dataset_mapper)

    eval_loader = copy.deepcopy(train_loader)
    eval_loader.dataset.split = val_split
    eval_loader.shuffle = False
    test_loader = copy.deepcopy(train_loader)
    test_loader.dataset.split = "test"
    eval_loader.shuffle = False

    return train_loader, eval_loader, test_loader