import os
import pathlib
import tarfile
import time
import urllib
from glob import glob
from typing import Sequence, Dict, Any, Union, Literal, Tuple

import numpy as np
import tifffile as tiff
import torch
from sklearn.model_selection import train_test_split

from pangaea.datasets.base import RawGeoFMDataset
from pangaea.datasets.utils import DownloadProgressBar

classes = {
    'StableForest':0,
    'StableNonForest':1,
    'ForestGain':2,
    'ForestLoss':3
}

class FAOFRACd(RawGeoFMDataset):
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
        gezs:list[int],
        n_timepoints:int,
        seed:int,
        test_size:float,
        val_size:float,
        shuffle:bool,
        **kwargs
    ):
        super(FAOFRACd, self).__init__(
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
        self.split = split
        self.bands = bands
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.img_size = img_size
        self.distribution = distribution
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.download_url = download_url
        self.auto_download = auto_download
        self.gezs = gezs
        self.n_timepoints = n_timepoints
        self.seed = seed
        self.test_size = test_size
        self.val_size=val_size
        self.shuffle = shuffle

        self.all_files, self.all_labels = self.get_all_files()

        ts_train, ts_test,label_train, label_test = train_test_split(self.all_files,self.all_labels,test_size=self.test_size,random_state=self.seed,shuffle=self.shuffle)
        ts_train, ts_val,label_train, label_val = train_test_split(ts_train,label_train,test_size=self.val_size,random_state=self.seed,shuffle=self.shuffle)

        if self.split == 'train':
            self.ts = ts_train
            self.labels = label_train
        elif self.split == 'test':
            self.ts = ts_test
            self.labels = label_test
        elif self.split == 'val':
            self.ts = ts_val
            self.labels = label_val
    def get_all_files(self):
            all_files = []
            all_labels = []
            for gez in self.gezs:
                folder = os.path.join(self.root_path,f'gez{gez}_hex_subsamples_tifs/')

                files = np.array(sorted(glob(folder + '/**/*.tif',recursive=True)))

                ts = np.reshape(files,(-1,2))
                gez_labels = np.stack(np.char.split(ts[:,0],sep='_'))[:,-4]

                all_files.append(ts)
                all_labels.append(gez_labels)
            
            all_files = np.concatenate(all_files,axis=0)
            all_labels = np.concatenate(all_labels,axis=0)

            return all_files, all_labels
    
    def __len__(self)->int:
        return len(self.ts)
    
    def __getitem__(self,idx:int) -> Dict[str,Union[torch.tensor, Any, str]]:
        item = self.ts[idx]
        label = self.labels[idx]

        tifs = [tiff.imread(im) for im in item]
        image = np.stack(tifs)
        image = image.astype(np.float32)  # Convert to float32
        image = torch.from_numpy(image).permute(3, 0, 1, 2)

        target = torch.tensor(classes[label])
        target = torch.nn.functional.one_hot(target,num_classes=self.num_classes)
        target = target.float()

        # invalid_mask = (image == 9999).any()
        # image[invalid_mask] = 0

        # images must have (C T H W) shape

        output = {
            "image": {
                "optical": image,
            },
            "target": target,
            "metadata": {},
        }

        return output



            
        @staticmethod
        def download(self):
            pass
