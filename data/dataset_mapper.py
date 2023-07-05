from typing import Any, Dict, List
import torch
import numpy as np

class DatasetMapperByVideo:

    def __init__(
        self,
        video_pipe,
        text_pipe,
    ) -> None:
        self.video_pipe = video_pipe
        self.text_pipe = text_pipe

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        video_ids = [item["video_id"] for item in batch]
        sentences = [sent for item in batch for sent in item["sentences"]]

        gt = torch.stack([torch.Tensor(_) / item["duration"] for item in batch for _ in item["timestamps"]], dim=0)

        video_result = self.video_pipe(video_ids=video_ids)
        text_result = self.text_pipe(sentences=sentences)

        batch_split_size = [len(item["sentences"]) for item in batch]

        return dict(**video_result, **text_result, batch_split_size=batch_split_size, gt=gt, batch=batch)

class DatasetMapperByPair:
    
    def __init__(self, video_pipe, text_pipe):
        self.video_pipe = video_pipe
        self.text_pipe = text_pipe
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        video_ids = [item["video_id"] for item in batch]
        sentences = [item["sentence"] for item in batch]
        
        gt = torch.stack([torch.Tensor(item["timestamp"]) / item["duration"] for item in batch], dim=0)

        video_result = self.video_pipe(video_ids=video_ids)
        text_result = self.text_pipe(sentences=sentences)

        return dict(**video_result, **text_result, gt=gt, batch=batch)
