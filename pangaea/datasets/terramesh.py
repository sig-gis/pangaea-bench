# Source: https://github.com/cloudtostreet/Sen1Floods11

import os

import geopandas
import numpy as np
import pandas as pd
import rasterio
import torch
import pathlib
from torch.utils.data import DataLoader
from pangaea.datasets.base import RawGeoFMDataset
from pangaea.datasets.utils import download_bucket_concurrently
from pangaea.datasets.terramesh_setup import build_terramesh_dataset, statistics, full_modality_set

from itertools import islice, count

import albumentations as A
from albumentations.pytorch import ToTensorV2
from terramesh import build_terramesh_dataset, Transpose, MultimodalTransforms, MultimodalNormalize, statistics
 
class TerraMesh(RawGeoFMDataset):
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
        modalities: list[str],
    ):
        """Initialize the TerraMesh dataset.
        Link: https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh

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
            e.g. {"s2_rgb": [b1_mean, ..., bn_mean], "s1_rtc": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2_rgb": [b1_std, ..., bn_std], "s1_rtc": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2_rgb": [b1_min, ..., bn_min], "s1_rtc": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2_rgb": [b1_max, ..., bn_max], "s1_rtc": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
            gcs_bucket (str): subset for downloading the dataset.
        """

        if modalities is None:
            self.modalities = full_modality_set
        else:
            self.modalities = modalities

        self.data_mean = statistics["mean"]
        self.data_std = statistics["std"]

        super(TerraMesh, self).__init__(
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

        self.distribution = distribution
        self.classes = classes
        self.img_size = img_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.download_url = download_url
        self.auto_download = auto_download



        shuf = True
        if split == "val":
            shuf = False


        val_transform = MultimodalTransforms(
            transforms=A.Compose([  # We use albumentations because of the shared transform between image modalities
                Transpose([1, 2, 0]),  # Convert data to channel last (expected shape from albumentations)
                MultimodalNormalize(mean=statistics["mean"], std=statistics["std"]),
                A.CenterCrop(224, 224),  # Use center crop in val split
                ToTensorV2(),  # Convert to tensor and back to channel first
                ],
                is_check_shapes=False,  # Not needed because of aligned data in TerraMesh
                additional_targets={m: "image" for m in modalities}  
                ),
            non_image_modalities=["__key__", "__url__"],  # Additional non-image keys
        )

        # If you pass multiple modalities, the modalities are returned using the modality names as keys
        self.terramind_dataset = build_terramesh_dataset(
            path=self.root_path, #"https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/", #self.root_path, #"https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/",  # Streaming or local path
            modalities=self.modalities,
            #return_metadata = True,
            shuffle=shuf,  # Set false for split="val"
            split=split,
            transform=val_transform,
            batch_size= 8 #From example on TerraMesh HF
        )


        # If you pass multiple modalities, the modalities are returned using the modality names as keys
        self.data_iter = DataLoader(self.terramind_dataset, batch_size=None) #, num_workers=4, persistent_workers=True, prefetch_factor=1)

 
        #for idx, batch in enumerate(self.terramind_dataset.__iter__()):
        #    print("DATALOADER_0", idx)

        #for batch in self.data_iter:
        #    print("DATALOADER_1", batch.keys())

        


        #self.data_iter = self.terramind_dataset.__iter__()

        self.length = sum(1 for _ in self.terramind_dataset.__iter__()) 


    def __len__(self):
        return self.length

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        return next(islice(self.data_iter, i, None))

    @staticmethod
    def download(self, silent=False):
        output_path = pathlib.Path(self.root_path)
        url = self.download_url

        try:
            os.makedirs(output_path, exist_ok=False)
        except FileExistsError:
            if not silent:
                print(
                    "TerraMesh dataset folder exists, skipping downloading dataset."
                )
            return

        pbar = DownloadProgressBar()

        try:
            urllib.request.urlretrieve(url, output_path , pbar)
        except urllib.error.HTTPError as e:
            print(
                "Error while downloading dataset: The server couldn't fulfill the request."
            )
            print("Error code: ", e.code)
            return
        except urllib.error.URLError as e:
            print("Error while downloading dataset: Failed to reach a server.")
            print("Reason: ", e.reason)
            return


