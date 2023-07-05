from kn_util.config import LazyCall as L
from kn_util.config import eval_str as E
from projects.contrastive_detr.modeling import ContrastiveDetr, TextEncoderPretrained, TextEncoderGloveTrm, VideoEncoderTrm, VideoDecoderDabTrm, Criterion, HungarianMatcher, ThresholdMatcher, HungarianMatcher, VideoDecoderTrm

model_cfg = dict(d_model=512, ff_dim=1024, nhead=8, dropout=0.1, vid_decoder_layers=3, vid_encoder_layers=3)

model = L(ContrastiveDetr)(
    txt_encoder=L(TextEncoderPretrained)(d_model="${model_cfg.d_model}", pretrained="distilbert-base-uncased"),
    vid_encoder=L(VideoEncoderTrm)(d_model="${model_cfg.d_model}",
                                   ff_dim="${model_cfg.ff_dim}",
                                   nhead="${model_cfg.nhead}",
                                   dropout="${model_cfg.dropout}",
                                   num_layers=3,
                                   vid_feat_dim=1024),
    vid_decoder=L(VideoDecoderDabTrm)(d_model="${model_cfg.d_model}",
                                   ff_dim="${model_cfg.ff_dim}",
                                   nhead="${model_cfg.nhead}",
                                   dropout="${model_cfg.dropout}",
                                   num_queries="${..num_queries}",
                                   num_layers=3),
    criterion=L(Criterion)(
        matcher=L(HungarianMatcher)(w_sim=0.0, w_iou=1.0),
        w_intra=1.0,
        w_inter=1.0,
        w_l1=1.0,
        w_iou=1.0,
        use_aux=True),
    d_model="${model_cfg.d_model}",
    num_queries=100,
    num_pattern=0)