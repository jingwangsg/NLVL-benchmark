from data.dataset_mapper import DatasetMapperByVideo
import torch
import torch.nn.functional as F

class DatasetMapperStandard(DatasetMapperByVideo):
    def __init__(self, video_pipe, text_pipe, normalize=True) -> None:
        super().__init__(video_pipe, text_pipe)
        self.normalize = normalize

    def __call__(self, batch):
        result = super().__call__(batch)
        if self.normalize:
            result["vid_feat"] = F.normalize(result["vid_feat"], dim=-1)
        
        return result