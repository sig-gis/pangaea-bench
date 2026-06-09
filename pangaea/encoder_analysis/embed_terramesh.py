import hashlib
import os as os
import pathlib
import pprint
import time

import numpy as np

import sys

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pangaea.datasets.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
from pangaea.datasets.terramesh import TerraMesh
from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder
from pangaea.engine.evaluator import Evaluator
from pangaea.engine.trainer import Trainer
from pangaea.utils.collate_fn import get_collate_fn
from pangaea.utils.logger import init_logger
from pangaea.utils.subset_sampler import get_subset_indices
from pangaea.datasets.terramesh import TerraMesh
from pangaea.encoders.terramind_encoder import terramind_v1_large

from pangaea.utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_final_model_ckpt_path,
    get_generator,
    seed_worker,
)

def get_exp_info(hydra_config: HydraConf) -> dict[str, str]:
    """Create a unique experiment name based on the choices made in the config.

    Args:
        hydra_config (HydraConf): hydra config.

    Returns:
        str: experiment information.
    """
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    cfg_hash = hashlib.sha1(
        OmegaConf.to_yaml(hydra_config).encode(), usedforsecurity=False
    ).hexdigest()[:6]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fm = choices["encoder"]
    decoder = choices["decoder"]
    ds = choices["dataset"]
    task = choices["task"]
    exp_info = {
        "timestamp": timestamp,
        "fm": fm,
        "decoder": decoder,
        "ds": ds,
        "task": task,
        "exp_name": f"{timestamp}_{cfg_hash}_{fm}_{decoder}_{ds}",
    }
    return exp_info


@hydra.main(version_base=None, config_path="../../configs", config_name="embed")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    # fix all random seeds
    fix_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = 'embed'
    exp_dir = './'
    exp_dir = pathlib.Path(cfg.work_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    task_name='Embed'
    logger_path = os.path.join(exp_dir,'embed.log')

    logger = init_logger(logger_path, rank=0)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    encoder: Encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger)
    logger.info("Built {}.".format(encoder.model_name))

    encoder = encoder.to(device)

    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities,return_meta=True)

    val_preprocessor = instantiate(
        cfg.preprocessing.val,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )

    raw_val_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="val")

    val_dataset = GeoFMDataset(
        raw_val_dataset, val_preprocessor, cfg.data_replicate
    )

    logger.info("Built {} dataset.".format(cfg.dataset.dataset_name))

    logger.info(
        f"Total number of validation patches: {len(val_dataset)}\n"
    )


    val_loader =  val_dataset.raw_dataset.terramind_dataset.__iter__()
 
    choices = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    out_dir = os.path.join(cfg.embed_dir,cfg.dataset.dataset_name,choices["encoder"],'val/')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok = True)


    fname_idx = 0
    for batch in val_loader:
        feat = {}

        batch_data = {}
        for modality, value in batch.items():
            if "__" in modality:
                continue
            batch_data[modality] = value.to(device, dtype=torch.float32)
        if not val_dataset.multi_temporal:
            with torch.no_grad():
                feat = {}
                for k,v in batch_data.items():
                    if "__" in k:
                        continue
                    feat[k] = encoder({k:v})

        for modality, value in feat.items():
            print(feat[modality][0].shape, len(feat[modality]))
            feat[modality] = value[-1]
            feat[modality] = feat[modality][0].cpu().detach().numpy()

  
        logit_out_fname = "embd_terramesh_" + str(fname_idx)  + ".npy" #Not shuffling for this experimentation - need a better method for future.
        np.savez(os.path.join(out_dir,logit_out_fname) ,**feat)


        #Saving crop info to allow for full reconstruction and uniform sampling across the same image in multiple runs
        if "crop" in batch:
            clip_info_out_fname = "crop_info_terramesh_" + str(fname_idx) + '.npy'
            for k, v in batch["image"].items():
                print(batch["crop"][k], k, batch["crop"]["target"])
            np.save(os.path.join(out_dir,clip_info_out_fname),batch["crop"])

        
        print(logit_out_fname, os.path.join(out_dir,logit_out_fname))
        fname_idx += 1


if __name__ == "__main__":
    main()
