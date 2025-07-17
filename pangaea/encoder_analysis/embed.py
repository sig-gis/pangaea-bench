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
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pangaea.datasets.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder
from pangaea.engine.evaluator import Evaluator
from pangaea.engine.trainer import Trainer
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

    # get preprocessor
    train_preprocessor = instantiate(
        cfg.preprocessing.train,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )
    val_preprocessor = instantiate(
        cfg.preprocessing.val,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )

    # get datasets
    raw_train_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="train")
    raw_val_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="val")

    train_dataset = GeoFMDataset(
        raw_train_dataset, train_preprocessor, cfg.data_replicate
    )
    val_dataset = GeoFMDataset(
        raw_val_dataset, val_preprocessor, cfg.data_replicate
    )

    logger.info("Built {} dataset.".format(cfg.dataset.dataset_name))

    logger.info(
        f"Total number of train patches: {len(train_dataset)}\n"
        f"Total number of validation patches: {len(val_dataset)}\n"
    )

    # get train val data loaders
    train_loader = DataLoader(
        train_dataset,
        # sampler=DistributedSampler(train_dataset),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        # persistent_workers=True causes memory leak
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=get_generator(cfg.seed),
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        # sampler=DistributedSampler(val_dataset),
        batch_size=cfg.test_batch_size,
        num_workers=cfg.test_num_workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        # generator=g,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Evaluation
    test_preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )

    # get datasets
    raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)

    test_loader = DataLoader(
        test_dataset,
        # sampler=DistributedSampler(test_dataset),
        batch_size=cfg.test_batch_size,
        num_workers=cfg.test_num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn,
    )


    choices = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    out_dir = os.path.join(cfg.embed_dir,cfg.dataset.dataset_name,choices["encoder"],'train/')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok = True)
    #embed train data

    for batch_idx, data in enumerate(train_loader):
        image, target, meta = data["image"], data["target"], data['metadata']

        

        image_fname = meta['image_filename'][0]

        image = {modality: value.to(device) for modality, value in image.items()}

        target = target.to(device)
        if encoder.multi_temporal:
            if not train_dataset.multi_temporal:
                with torch.no_grad():
                    feat = encoder(image)
                if encoder.multi_temporal_output:
                    feat = [f.squeeze(-3) for f in feat]
            else:
                with torch.no_grad():
                    feat = encoder(image)
        else:
            if not train_dataset.multi_temporal:
                with torch.no_grad():
                    feat = encoder({k: v[:, :, 0, :, :] for k, v in image.items()})
            else:
                feats = []
                for i in range(train_dataset.multi_temporal):
                    with torch.no_grad():
                        feats.append(
                            torch.stack(encoder({k: v[:, :, i, :, :] for k, v in image.items()}),dim=0)
                        )
                    feat = torch.stack(feats,dim=2)

        feat = feat[-1]
        feat = feat[0].cpu().detach().numpy()


        logit_out_fname = 'embd_' + image_fname[:-3] + 'npy'
        print(logit_out_fname)
        np.save(os.path.join(out_dir,logit_out_fname),feat)


    out_dir = os.path.join(cfg.embed_dir,cfg.dataset.dataset_name,choices["encoder"],'test/')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok = True)
    
    #embed test data
    for batch_idx, data in enumerate(test_loader):
        if "filename" in data:
            image, target, image_fname, meta = data["image"], data["target"],data["filename"], data['metadata']
            image_fname = image_fname[0]
        else:
            image, target, meta = data["image"], data["target"], data['metadata']
            image_fname = meta['image_filename'][0]



        image = {modality: value.to(device) for modality, value in image.items()}

        target = target.to(device)
        if encoder.multi_temporal:
            if not train_dataset.multi_temporal:
                with torch.no_grad():
                    feat = encoder(image)
                if encoder.multi_temporal_output:
                    feat = [f.squeeze(-3) for f in feat]
            else:
                with torch.no_grad():
                    feat = encoder(image)
        else:
            if not train_dataset.multi_temporal:
                with torch.no_grad():
                    feat = encoder({k: v[:, :, 0, :, :] for k, v in image.items()})
            else:
                feats = []
                for i in range(train_dataset.multi_temporal):
                    with torch.no_grad():
                        feats.append(
                            torch.stack(encoder({k: v[:, :, i, :, :] for k, v in image.items()}),dim=0)
                        )
                feat = torch.stack(feats,dim=2)


        feat = feat[-1]
        feat = feat[0].cpu().detach().numpy()


        logit_out_fname = 'embd_' + os.path.splitext(image_fname)[0] + '.npy'
        print(logit_out_fname)

        np.save(os.path.join(out_dir,logit_out_fname),feat)



if __name__ == "__main__":
    main()
