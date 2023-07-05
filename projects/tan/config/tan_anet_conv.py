from projects.tan.config.base_model import model
from kn_util.config import LazyCall as L
from kn_util.config import eval_str
from config.common.basic import paths, runtime
from data.dataset import DatasetByVideoQueryPair
from data.datapipe import TextPipeGlove, VideoPipeHDF5
from projects.tan.dataset_mapper import DatasetMapperTAN
from config.common.dataloader import build_dataloaders
from config.common.evaluater import build_evaluater_nlvl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

data = dict(dataset="activitynet", video_length=256, feat_type="i3d")
train = dict(num_epochs=100, batch_size=32, gradient_accumulation_steps=1, interval_log_train=0.1, interval_eval=1.0)

_hdf5_file = "${paths.dataset}/${data.feat_type}.hdf5"
_dataset = L(DatasetByVideoQueryPair)(annot_dir="${paths.dataset}", dataset="${data.dataset}", split="train")
_dataset_mapper = L(DatasetMapperTAN)(video_pipe=L(VideoPipeHDF5)(hdf5_file=_hdf5_file,
                                                                  to_length="${data.video_length}",
                                                                  pad_or_sample="sample"),
                                      text_pipe=L(TextPipeGlove)(cache_dir="${paths.data}/.cache"),
                                      num_clips=eval_str("${data.video_length}//${model.video_encoder.stride}"))
train_loader, eval_loader, test_loader = build_dataloaders(dataset=_dataset, dataset_mapper=_dataset_mapper)

train_evaluater, eval_evaluater, test_evaluater = build_evaluater_nlvl(losses=["loss"])

optimizer = L(AdamW)(lr=1e-4, weight_decay=0.0)
lr_scheduler = L(ReduceLROnPlateau)(mode="min", factor=0.8, patience=20, verbose=True)
train["lr_schedule_monitor"] = "eval/loss"