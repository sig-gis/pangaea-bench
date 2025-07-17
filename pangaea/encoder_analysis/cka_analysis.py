import hashlib
import os as os
import pathlib
import pprint
import time

import numpy as np

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from hydra import compose, initialize

from omegaconf import DictConfig, OmegaConf

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pangaea.datasets.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
from pangaea.encoders.base import Encoder
from pangaea.utils.collate_fn import get_collate_fn
from pangaea.utils.logger import init_logger
from pangaea.utils.subset_sampler import get_subset_indices
from pangaea.utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_final_model_ckpt_path,
    get_generator,
    seed_worker,
)

import math
import zarr
import umap
import copy
from torch_cka import CKA
import sys

def run_pairwise_cka_analysis(encoder1, encoder2, loader1, loader2, out_dir):
 
    model1_layers = []

    full_layer_set = encoder1.named_children()
    submodule_count = len(list(full_layer_set))
    ind = 0
    for name, _ in full_layer_set:
        if submodule_count - ind < 1:
            model1_layers.append(name) 

    model2_layers = []
    full_layer_set = encoder2.named_children()
    submodule_count = len(list(full_layer_set))
    ind = 0
    for name, _ in full_layer_set:
        if submodule_count - ind < 1:
            model2_layers.append(name) 


    cka = CKA(encoder1, encoder2,
        model1_name=encoder1.model_name, model2_name=encoder2.model_name,
        model1_layers=model1_layers, model2_layers=model2_layers, device='cpu')
 
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok = True)

    cka.compare(loader1, loader2) #, only_compare_diagonals=True)
    cka.plot_results(save_path=os.path.join(out_dir, encoder1.model_name + "_" + encoder2.model_name+ "_compare.png"))

def get_init_config(overrides):

    with initialize(version_base=None, config_path="../../configs/"):
        cfg = compose(config_name="cka", overrides=overrides)

    return cfg


def main(): #cfg: DictConfig) -> None:
 
        model_names = [
        "croma_optical",
        "dofa",
        "gfmswin",
        "prithvi",
        "satlasnet_si",
        "scalemae",
        "spectralgpt",
        "ssl4eo_data2vec",
        "ssl4eo_dino",
        "ssl4eo_mae_optical",
         "ssl4eo_moco",
         "unet_encoder",
         "terramind_large",
         "resnet50_scratch",
         "resnet50_pretrained",
         "vit_scratch",
         "vit"
        ]



        input_resize_encoders = ["croma_optical", "dofa", "prithvi", "satlasnet_si", "scalemae"]
 

        cfg = get_init_config(["dataset=hlsburnscars"])

        fix_seed(cfg.seed)  
 
        exp_name = 'cka_analysis'
        exp_dir = './'
        exp_dir = pathlib.Path(cfg.work_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        task_name='CKA'
        logger_path = os.path.join(exp_dir,'cka.log')

        logger = init_logger(logger_path, rank=0)
        logger.info("============ Initialized logger ============")
        logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
        logger.info("The experiment is stored in %s\n" % exp_dir)

        out_dir = cfg.out_dir

        print("HERE")
        for i in range(len(model_names)-1):
    
            #Decoder is not needed for experiment, but required for configuration
            decoder = "seg_upernet"
            if "unet" in model_names[i]:
                decoder = "seg_unet"

            preprocess_scheme = "seg_resize"
            if model_names[i] in input_resize_encoders: 
                preprocess_scheme = "seg_resize_input_layer"
            #else:
            #    continue

            batch_size = cfg.test_batch_size
            num_workers = cfg.test_num_workers

            input_bands = cfg.dataset.bands
            print(cfg.dataset.bands)
            print(cfg.dataset.img_size)


            print(model_names[i])
            GlobalHydra.instance().clear()
            
            encoder1_config = None
          

            #"encoder=" + model_names[i], \
            with initialize(version_base=None, config_path="../../configs/"):
                overrides = overrides=["dataset=hlsburnscars", "encoder=" + model_names[i],\
                            "decoder="+decoder, "preprocessing="+preprocess_scheme] #TODO
                encoder1_config = compose(config_name="cka_encoder", overrides=overrides)
                #encoder1_config = OmegaConf.create(encoder1_config)
 
                raw_test_dataset: RawGeoFMDataset = instantiate(encoder1_config.dataset, split="test")
                encoder1: Encoder = instantiate(encoder1_config.encoder)
                encoder1.load_encoder_weights(logger)
                logger.info("Built {}.".format(encoder1.model_name))

                #Evaluation
                test_preprocessor_1 = instantiate(
                        encoder1_config.preprocessing.test,
                        dataset_cfg=encoder1_config.dataset, #TODO choices["dataset"],
                        encoder_cfg=encoder1_config.encoder,
                        _recursive_=False,
                )
     

                modalities = list(encoder1.input_bands.keys())
                collate_fn = get_collate_fn(modalities,return_meta=True)
 
                # get datasets
                test_dataset_1 = GeoFMDataset(raw_test_dataset, test_preprocessor_1)
     
                loader1 = DataLoader(
                    test_dataset_1,
                    # sampler=DistributedSampler(test_dataset),
                    batch_size= batch_size,
                    num_workers= num_workers, 
                    pin_memory=True,
                    persistent_workers=False,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
    
                for j in range(i+1, len(model_names)):
                    #GlobalHydra.instance().clear()
 
                    decoder = "seg_upernet"
                    if "unet" in model_names[i]:
                        decoder = "seg_unet"
     
                    #cfg2 = copy.deepcopy(cfg)
                    encoder2_config = None

                    preprocess_scheme = "seg_resize"
                    if model_names[j] in input_resize_encoders:
                        preprocess_scheme = "seg_resize_input_layer"
                    #else:
                    #    continue
 
                    if True: 
                        overrides = overrides=["dataset=hlsburnscars", "encoder=" + model_names[j], \
                            "decoder="+decoder, "preprocessing="+preprocess_scheme]
                        encoder2_config = compose(config_name="cka_encoder", overrides=overrides)
 

                        encoder2: Encoder = instantiate(encoder2_config.encoder)
                        encoder2.load_encoder_weights(logger)
                        logger.info("Built {}.".format(encoder2.model_name))
    

                        test_preprocessor_2 = instantiate(
                            encoder2_config.preprocessing.test,
                            dataset_cfg=encoder2_config.dataset, #TODO choices["dataset"],
                            encoder_cfg=encoder2_config.encoder,
                            _recursive_=False,
                        )
    
                        modalities = list(encoder2.input_bands.keys())
                        collate_fn = get_collate_fn(modalities,return_meta=True)  

                        test_dataset_2 = GeoFMDataset(raw_test_dataset, test_preprocessor_2)
    
                        loader2 = DataLoader(        
                            test_dataset_2,
                            # sampler=DistributedSampler(test_dataset),
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=False,
                            drop_last=False,
                            collate_fn=collate_fn,
                        )


                        run_pairwise_cka_analysis(encoder1, encoder2, loader1, loader2, out_dir)
                    #GlobalHydra.instance().clear()


   
 

if __name__ == "__main__":
    main()
