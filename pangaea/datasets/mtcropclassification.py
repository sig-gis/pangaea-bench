import os
import pathlib
import tarfile
import time
import urllib
from glob import glob
from typing import Sequence, Tuple

import numpy as np
import tifffile as tiff
import torch
from sklearn.model_selection import train_test_split

from pangaea.datasets.base import RawGeoFMDataset
from pangaea.datasets.utils import DownloadProgressBar

from einops import rearrange

class MTCropClassification(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
    ):
        """Initialize the MultiTemporalCropClassification dataset.
        Link: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars

        Args:
            split (str): split of the dataset (train, val, test).
            dataset_name (str): dataset name.
            multi_modal (bool): if the dataset is multi-modal.
            multi_temporal (int): number of temporal frames.
            root_path (str): root path of the dataset.
            classes (list): classes of the dataset.
            num_classes (int): number of classes.
            ignore_index (int): index to ignore for metrics and loss.
            img_size (int): size of the image.
            bands (dict[str, list[str]]): bands of the dataset.
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality.
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
        """

        super(MTCropClassification, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        self.root_path = root_path
        self.classes = classes
        self.split = split
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.img_size = img_size
        self.distribution = distribution
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.download_url = download_url
        self.auto_download = auto_download

        self.split_mapping = {
            "train": "training_chips",
            "val": "training_chips",
            "test": "validation_chips",
        }

        all_files = sorted(
            glob(
                os.path.join(
                    self.root_path, self.split_mapping[self.split], "*merged.tif"
                )
            )
        )
        all_targets = sorted(
            glob(
                os.path.join(
                    self.root_path, self.split_mapping[self.split], "*mask.tif"
                )
            )
        )

        if self.split != "test":
            split_indices = self.get_train_val_split(all_files)
            if self.split == "train":
                indices = split_indices["train"]
            else:
                indices = split_indices["val"]
            self.image_list = [all_files[i] for i in indices]
            self.target_list = [all_targets[i] for i in indices]
        else:
            self.image_list = all_files
            self.target_list = all_targets

    @staticmethod
    def get_train_val_split(all_files) -> Tuple[Sequence[int], Sequence[int]]:
        # Fixed stratified sample to split data into train/val.
        # This keeps 90% of datapoints belonging to an individual event in the training set and puts the remaining 10% in the validation set.
        train_idxs, val_idxs = train_test_split(
            np.arange(len(all_files)),
            test_size=0.1,
            random_state=23,
        )
        return {"train": train_idxs, "val": val_idxs}

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = tiff.imread(self.image_list[index])
        image = image.astype(np.float32)  # Convert to float32
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = rearrange(image,'(c t) h w -> c t h w',c=int(image.shape[0] / self.multi_temporal))

        target = tiff.imread(self.target_list[index])
        target = target.astype(np.int64)  # Convert to int64 (since it's a mask)
        target = torch.from_numpy(target).long()

        invalid_mask = image == 9999
        image[invalid_mask] = 0

        # images must have (C T H W) shape
        # image = image.unsqueeze(1)
        output = {
            "image": {
                "optical": image,
            },
            "target": target,
            "metadata": {
                'image_filename':self.image_list[index],
                'target_filename':self.target_list[index]
                },
        }

        return output

    @staticmethod
    def download(self, silent=False):
        pass

