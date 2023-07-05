from typing import Any
from transformers import AutoTokenizer
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from kn_util.data import sample_sequence_general, pad_sequence_general
from kn_util.data.processor import GloveTokenizeProcessor
from kn_util.basic import global_get


class TextPipePretrained:
    """
    1. tokenize sentences
    2. get indices and mask
    """

    def __init__(self, pretrained) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def __call__(self, sentences):
        output = self.tokenizer(sentences, padding=True, return_tensors="pt")
        txt_inds = output.input_ids
        txt_mask = output.attention_mask.bool()
        return dict(txt_inds=txt_inds, txt_mask=txt_mask)


class TextPipeGlove:

    def __init__(
        self,
        glove="glove.6B.300d",
        vocab_file=None,
        global_vocab_key="glove",
        tokenizer="split",
        cache_dir=None,
    ) -> None:
        self.tokenizer = GloveTokenizeProcessor(glove=glove,
                                                global_vocab_key=global_vocab_key,
                                                vocab_file=vocab_file,
                                                tokenizer=tokenizer,
                                                cache_dir=cache_dir,
                                                to_indices=True,
                                                to_embeddings=False)
        self.global_vocab_key = global_vocab_key

    def __call__(self, sentences):

        txt_inds_list = [self.tokenizer(sentence).indices for sentence in sentences]
        txt_inds, txt_mask = pad_sequence_general(txt_inds_list,
                                                  fill_value=global_get(self.global_vocab_key).stoi["<pad>"],
                                                  return_mask=True)
        txt_inds = torch.from_numpy(np.stack(txt_inds, axis=0))
        txt_mask = torch.from_numpy(np.stack(txt_mask, axis=0))

        return dict(txt_inds=txt_inds, txt_mask=txt_mask)


class VideoPipeHDF5:
    """
    1. Read video features from hdf5 file.
    2. sample video features to a fixed length.
    """

    def __init__(self, hdf5_file, to_length=128, pad_or_sample="pad") -> Any:
        self.h5 = h5py.File(hdf5_file, "r")
        self.to_length = to_length
        self.pad_or_sample = pad_or_sample

    def __call__(self, video_ids):
        eps = 1e-12
        vid_feat = [np.array(self.h5[video_id]) for video_id in video_ids]
        vid_feat = [x / (np.linalg.norm(x, axis=1, keepdims=True) + eps) for x in vid_feat]

        if self.pad_or_sample == "sample":
            vid_feat_sampled = [sample_sequence_general(feat, seq_len=self.to_length) for feat in vid_feat]
            vid_feat = torch.from_numpy(np.stack(vid_feat_sampled, axis=0))
        elif self.pad_or_sample == "pad":
            vid_feat_pad = pad_sequence_general(vid_feat, fill_value=0.0, axis=0)
            vid_feat = torch.from_numpy(np.stack(vid_feat_pad, axis=0))

        return dict(vid_feat=vid_feat)