from torch.utils.data import Dataset
from kn_util.basic import load_json, load_csv
import os.path as osp
import h5py
from loguru import logger


class ValidaterHDF5:

    def __init__(self, hdf5) -> None:
        self.hdf5 = h5py.File(hdf5, "r")

    def __call__(self, video_ids):
        valid_ids = [video_id for video_id in video_ids if video_id in self.hdf5]
        self.hdf5.close()
        logger.info("=> {} valid videos".format(len(valid_ids)))
        return valid_ids


class ValidaterFile:

    def __init__(self, path_template="{}.mp4") -> None:
        self.template = path_template

    def __call__(self, video_ids):
        valid_ids = [video_id for video_id in video_ids if osp.exists(self.template.format(video_id))]
        logger.info("=> {} valid videos".format(len(valid_ids)))
        return valid_ids


class DatasetNLVLBase(Dataset):

    def __init__(self, annot_dir, dataset="activitynet", split="train", validater=None):
        self.validater = validater
        self.dataset = self.prepare_data(annot_dir, split, dataset=dataset)

    def prepare_data_charades(self, annot_dir, split):
        raise NotImplementedError

    def prepare_data_activitynet(self, annot_dir, split):
        raise NotImplementedError

    def prepare_data_tacos(self, annot_dir, split):
        raise NotImplementedError

    def prepare_data(self, annot_dir, split, dataset):
        call_dict = dict(
            charades=self.prepare_data_charades,
            activitynet=self.prepare_data_activitynet,
            tacos=self.prepare_data_tacos,
        )

        return call_dict[dataset](annot_dir, split)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class DatasetByVideoQueryPair(DatasetNLVLBase):

    def __init__(self, annot_dir, dataset="activitynet", split="train", validater=None):
        super().__init__(annot_dir, dataset, split, validater)

    def prepare_data_activitynet(self, annot_dir, split):
        filepath = osp.join(annot_dir, f"{split}.json")
        json_dict = load_json(filepath)
        if self.validater is not None:
            video_ids = self.validater(list(json_dict.keys()))
        else:
            video_ids = list(json_dict.keys())

        dataset = []
        for video_id in video_ids:
            video_info = json_dict[video_id]
            for sentence, timestamp in zip(video_info["sentences"], video_info["timestamps"]):
                if not (0 <= timestamp[0] < timestamp[1] <= video_info["duration"]):
                    continue
                item = dict()
                item["video_id"] = video_id
                item["duration"] = video_info["duration"]
                item["sentence"] = (sentence)
                item["timestamp"] = (timestamp)
                dataset.append(item)

        return dataset


class DatasetByVideo(DatasetNLVLBase):

    def __init__(self, annot_dir, dataset="activitynet", split="train", validater=None):
        super().__init__(annot_dir, dataset, split, validater)

    def prepare_data_activitynet(self, annot_dir, split):
        filepath = osp.join(annot_dir, f"{split}.json")
        json_dict = load_json(filepath)
        if self.validater is not None:
            video_ids = self.validater(list(json_dict.keys()))
        else:
            video_ids = list(json_dict.keys())

        dataset = []
        for video_id in video_ids:
            item = dict()
            video_info = json_dict[video_id]
            item["video_id"] = video_id
            item["duration"] = video_info["duration"]
            item["sentences"] = []
            item["timestamps"] = []
            for sentence, timestamp in zip(video_info["sentences"], video_info["timestamps"]):
                if 0 <= timestamp[0] < timestamp[1] <= item["duration"]:
                    item["sentences"].append(sentence)
                    item["timestamps"].append(timestamp)
            if len(item["sentences"]) > 0:
                dataset.append(item)

        return dataset
