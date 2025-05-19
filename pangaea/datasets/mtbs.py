# from osgeo import gdal
# from osgeo.gdalconst import GA_Update

# # ... and suppress errors
# gdal.PushErrorHandler('CPLQuietErrorHandler')
# gdal.SetConfigOption('CPL_LOG','mnt/log/')

import os
import pathlib
import tarfile
import time
import urllib
import glob
from typing import Sequence, Tuple
import datetime
import itertools

import rasterio

import numpy as np
import tifffile as tiff
import torch
from pangaea.datasets.base import RawGeoFMDataset
from pangaea.datasets.utils import DownloadProgressBar

from einops import rearrange

crs = 'EPSG:3857'

sample_regions = [
    'Northern_Cascades',
    'Sierra_Nevada_Mountain_Range',
    'Snake_River_Plain',
    'Southern_Appalachia'
]

def convert_date(path):
    datestr = path.split('_')[-1][:-4]

    year = int(datestr[0:4])
    mon = int(datestr[4:6])
    day = int(datestr[6:])

    dt = datetime.datetime(year,mon,day)

    return dt


class MTBSBurnSeverity(RawGeoFMDataset):
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
        super(MTBSBurnSeverity, self).__init__(
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

        self.root_path = os.path.join(root_path,'mtbs_fires')

        gj_files = glob.glob(os.path.join(self.root_path,'*.geojson'))

        self.meta = dict(zip(sample_regions,gj_files))

        sample_regions_images = dict()
        sample_regions_annotations = dict()

        for region in sample_regions:
            region_images = sorted(glob.glob(os.path.join(self.root_path,'data',self.meta[region].split('/')[-1][:-8],'images') + '/*.tif'))
            region_labels = sorted(glob.glob(os.path.join(self.root_path,'data',self.meta[region].split('/')[-1][:-8],'labels') + '/*.tif'))

            sample_regions_images[region] = region_images
            sample_regions_annotations[region] = region_labels

        full_dataset_images = np.array(sorted(list(itertools.chain.from_iterable(list(sample_regions_images.values())))))
        v_convert_date = np.vectorize(convert_date)
        full_dataset_dates = v_convert_date(full_dataset_images) 

        self.full_dataset_images = np.reshape(full_dataset_images,(int(len(full_dataset_images) / 4),-1))
        self.full_dataset_dates = np.reshape(full_dataset_dates,(int(len(full_dataset_dates) / 4),-1))

        self.full_dataset_annotations =  np.array(sorted(list(itertools.chain.from_iterable(list(sample_regions_annotations.values())))))

        self.split_mapping = {
            'train':np.load(os.path.join(self.root_path,'splits','train.npy')),
            'val':np.load(os.path.join(self.root_path,'splits','val.npy')),
            'test':np.load(os.path.join(self.root_path,'splits','test.npy'))
        }

            
        self.image_files = self.full_dataset_images[self.split_mapping[self.split]]
        self.dates = self.full_dataset_dates[self.split_mapping[self.split]]
        self.annotations = self.full_dataset_annotations[self.split_mapping[self.split]]

    def __len__(self):
        return len(self.image_files)
    def __getitem__(self,index):
        im_paths = self.image_files[index]
        annotation_path = self.annotations[index]

        rasters = [rasterio.open(image_path,dtype=np.float64) for image_path in im_paths]
        latlons = [val for val in rasters[0].bounds]

        frames = [raster.read()[:,:512,:512] for raster in rasters]

        image = np.stack(frames,axis=0)
        image = image[:-1,:,:,:]
        image = image.astype(np.float32)  # Convert to float32
        image = torch.from_numpy(image).permute(1,0,2,3)

        annotation = rasterio.open(annotation_path,dtype=np.float64)
        annotation = annotation.read()[0,:512,:512]
        target = annotation.astype(np.int64)  # Convert to int64 (since it's a mask)
        target = torch.from_numpy(target).long()

        output = {
            'image':{
                'optical':image
            },
            'target':target,
            'metadata':{'image_filename':im_paths[0].split('/')[-1]}
        }

        return output 
    
    @staticmethod
    def download(self, silent=False):
        pass


        