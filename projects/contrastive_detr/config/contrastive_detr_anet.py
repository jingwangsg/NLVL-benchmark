from kn_util.config import LazyCall as L
from data.dataset import DatasetByVideo
from data.datapipe import TextPipePretrained, VideoPipeHDF5
import os.path as osp
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from config.common.basic import paths, runtime
from config.common.dataloader import build_dataloaders
from config.common.evaluater import build_evaluater_nlvl
from projects.contrastive_detr.dataset_mapper import DatasetMapperStandard
from projects.contrastive_detr.config.base_model import model, model_cfg

data = dict(dataset="activitynet", feat_type="i3d", video_length=128)
train = dict(num_epochs=10, batch_size=16, gradient_accumulation_steps=1, interval_log_train=0.1, interval_eval=1.0)

_annot_dir = osp.join("${paths.data}", "${data.dataset}")
_hdf5_file = osp.join("${paths.data}", "${data.dataset}", "${data.feat_type}.hdf5")
_dataset = L(DatasetByVideo)(annot_dir=_annot_dir, dataset="${data.dataset}")
_dataset_mapper = L(DatasetMapperStandard)(video_pipe=L(VideoPipeHDF5)(hdf5_file=_hdf5_file,
                                                                       to_length="${data.video_length}"),
                                           text_pipe=L(TextPipePretrained)(pretrained="distilbert-base-uncased"))
train_loader, eval_loader, test_loader = build_dataloaders(dataset=_dataset, dataset_mapper=_dataset_mapper)


def losses_lazy(use_aux=True, num_layers=3):
    lvls = range(num_layers) if use_aux else [0]
    losses = ["intra", "inter", "iou", "l1"]
    return ["loss"] + [f"loss_{loss_type}_{lvl}" for loss_type in losses for lvl in lvls]


_losses = L(losses_lazy)(use_aux="${model.criterion.use_aux}", num_layers="${model.vid_decoder.num_layers}")
train_evaluater, eval_evaluater, test_evaluater = build_evaluater_nlvl(losses=_losses)

optimizer = L(AdamW)(params=None, lr=1e-4)
lr_scheduler = L(MultiStepLR)(optimizer=None, milestones=[50, 75])
# checkpointer = L(CheckPointer)(monitor="eval/IoU=0.7@R1", mode="max", work_dir="${paths.work_dir}")

lr_schema = {
    "txt_encoder": 0.01,
}