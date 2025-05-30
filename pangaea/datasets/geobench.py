import ast
import json
import h5py
import numpy as np
import os
import pickle
import torch

from geobench.dataset import Band, Sample

from pangaea.datasets.base import RawGeoFMDataset


PARTITION_MAP = {
    "train": "train",
    "val": "valid",
    "test": "test",
}
DATASET_NAMES = (
    "m-bigearthnet",
    "m-brick_kiln",
    "m-eurosat",
    "m-forestnet",
    "m-pv4ger",
    "m-so2sat",
    "m-cashew_plant",
    "m-chesapeake",
    "m-neontree",
    "m-nz_cattle",
    "m-pv4ger_seg",
    "m-sa_crop_type "
)
S2_BAND_NAMES_MAP = {
    "B1": "01 - Coastal aerosol",
    "B2": "02 - Blue",
    "B3": "03 - Green",
    "B4": "04 - Red",
    "B5": "05 - Vegetation Red Edge",
    "B6": "06 - Vegetation Red Edge",
    "B7": "07 - Vegetation Red Edge",
    "B8": "08 - NIR",
    "B8A": "08A - Vegetation Red Edge",
    "B9": "09 - Water vapour",
    "B10": "10 - SWIR - Cirrus",
    "B11": "11 - SWIR",
    "B12": "12 - SWIR",
}
BAND_NAMES_MAP = {
    "B2": "Blue",
    "B3": "Green",
    "B4": "Red",
    "B8": "NearInfrared",
}



class GeoBench(RawGeoFMDataset):
    """"""
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
        dataset_cat: str,
        partition_lim: float
    ):
        """Initializes the dataset.

        Args:
            split (str): split of the dataset (train, val, test)
            dataset_name (str): dataset name
            multi_modal (bool): whether the dataset is multi_modal
            multi_temporal (int): number of temporal frames
            root_path (str): root path of the dataset
            classes (list): dataset classes names
            num_classes (int): number of classes
            ignore_index (int): index to ignore
            img_size (int): dataset's image size
            bands (dict[str, list[str]]): bands of the dataset
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
        super(GeoBench, self).__init__(
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

        if not os.path.exists(self.root_path):
            raise FileNotFoundError("file not found; download and upload")
        self.dataset_cat = dataset_cat
        self.partition_lim = f"{partition_lim:.02f}"

        self._load_partition()

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Raises:
            NotImplementedError: raise if the method is not implemented

        Returns:
            int: length of the dataset
        """
        raise NotImplementedError

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Returns the i-th item of the dataset.

        Args:
            i (int): index of the item

        Raises:
            NotImplementedError: raise if the method is not implemented

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
            {"image":
                {
                "optical": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                 "sar": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                 },
            "target": torch.Tensor of shape (H W) of type torch.int64 for segmentation, torch.float for
            regression datasets.,
             "metadata": dict}.
        """
        raise NotImplementedError

    def _load_partition(self) -> None:
        """Load a partition."""
        self.dataset_dir = os.path.join(self.root_path,
                                        self.dataset_cat,
                                        self.dataset_name)
        self.dataset_partition = json.loads(f"{self.partition_lim}_partition.json")
        self.dataset = self.dataset_partition[PARTITION_MAP[self.partition]]

    def _load_sample_hdf5(sample_path: os.PathLike, band_names=None, label_only=False):
        """Load hdf5 sample.

        Args:
            sample_path: path to the sample
            band_names: list of bandnames to return from sample
            label_only: whether or not to only return the label

        Returns:
            loaded sample
        """
        with h5py.File(sample_path, "r") as fp:
            attr_dict = pickle.loads(ast.literal_eval(fp.attrs["pickle"]))
            band_names = attr_dict.get("bands_order", fp.keys())
            bands = []
            label = None
            for band_name in band_names:
                if label_only and not band_name.startswith("label"):
                    continue
                h5_band = fp[band_name]

                band = Band(data=np.array(h5_band), **attr_dict[band_name])
                if band_name.startswith("label"):
                    label = band
                else:
                    bands.append(band)
            if label is None:
                label = attr_dict["label"]

            sample = Sample(bands=bands, label=label, sample_name=sample_path.stem)

            return sample
