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

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pangaea.datasets.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
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

import math
import zarr
import umap
import joblib
import rasterio

from sklearn.preprocessing import MinMaxScaler

import scipy
from scipy.spatial.distance import pdist, squareform


def knn_graph(w, k, symmetrize=True, metric='cosine'):
    '''
    :param w: A weighted affinity graph of shape [N, N] or 2-d array 
    :param k: The number of neighbors to use
    :param symmetrize: Whether to symmetrize the resulting graph
    :return: An undirected, binary, KNN graph of shape [N, N]
    '''
    w_shape = w.shape
    if w_shape[0] != w_shape[1]:
        w = np.array(squareform(pdist(w, metric=metric)))

    neighborhoods = np.argsort(w, axis=1)[:, -(k+1):-1]
    A = np.zeros_like(w)
    for i, neighbors in enumerate(neighborhoods):
        for j in neighbors:
            A[i, j] = 1
            if symmetrize:
                A[j, i] = 1
    return A

def build_knn_graph(embed, out_fname):
    if embed.ndim < 3:
        k=int(np.log(embed.shape[0]))
    else:
        k=int(np.log(embed.shape[0]* embed.shape[1]))

    knn_graph_out =  knn_graph(embed, k=k, symmetrize=True, metric='cosine')
    zarr.save(out_fname, knn_graph_out)


def get_indices(cfg, image_fname, target, indices_dir) -> None:
 
    #Get or set indices used to subset embeddings from individual files
    index_fname = os.path.join(indices_dir, os.path.splitext(image_fname)[0] + ".indices.zarr")

    final_inds = None
    n_classes = cfg.dataset.num_classes
    ignore_class = cfg.dataset.ignore_index

    if not os.path.exists(index_fname):
        sub_inds = []

        for i in range(0, n_classes+1):

            if i == ignore_class:
                continue

            sub_inds = np.where(target == i)[0]
            if len(sub_inds) < 1:
                continue

            #Ensure each class is represented - can add stratification later
            if len(sub_inds)  > cfg.label_file_subset:
                selection_inds  = np.random.choice(len(sub_inds), size=cfg.label_file_subset, replace=False)
                sub_inds = sub_inds[selection_inds]

            if final_inds is None:
                 final_inds = np.array(sub_inds)
            else:
                final_inds = np.concatenate((final_inds, sub_inds), axis=0)


        zarr.save(index_fname, final_inds)
    else:
        final_inds = zarr.load(index_fname)

    return final_inds



def rescale_embed(embed, image_shape, device):
 
     ind = 0
     if embed.ndim > 3:
         ind = 1


     if int(math.sqrt(embed.shape[ind])) // 8 > 3:
  
         rescale_factor = int(math.sqrt(embed.shape[ind])) // 8


         while embed.shape[ind] % rescale_factor**2 > 0:
             rescale_factor = rescale_factor + 1

         ps = torch.nn.PixelShuffle(rescale_factor) #.to(device)
         embed = torch.from_numpy(embed).to(device)

         embed = ps(embed)
     else:
         embed = torch.from_numpy(embed).to(device)

     if embed.ndim < 4:
         embed = torch.unsqueeze(embed, dim=0)
     else:
         embed = torch.unsqueeze(torch.flatten(embed, start_dim=0, end_dim=1), dim=0)
         

     #Assumption is currently square images + tiles 
     #Adjusting to account for potential off-by-ones
     if embed.shape[-1] != image_shape:
         embed = F.interpolate(embed, size=(image_shape, image_shape), mode='nearest')

     embed = torch.permute(embed, (0,2,3,1)).flatten(start_dim=0, end_dim=2)
 
     if device == "cuda":
         embed = embed.detach().cpu().numpy()

 
     return embed

def train_and_gen_projection(embed, out_dir, cfg):
   
    reducer = None
    scaler = None
    reducer_fname = os.path.join(out_dir, 'umap_model.joblib')
    #scaler_fname = os.path.join(out_dir, 'umap_pre_scaler.joblib')

    #if not os.path.exists(scaler_fname):
    #    scaler = MinMaxScaler()
    #    embed = scaler.fit_transform(embed)
    #    joblib.dump(scaler, scaler_fname)
    #else:
    #    scaler = joblib.load(scaler_fname)
    #    embed = scaler.transform(embed)    
 
    if not os.path.exists(reducer_fname):
        print("Training UMAP and projecting data", embed.shape)
        reducer = umap.UMAP(metric="cosine", n_neighbors=cfg.umap_n_neighbors, \
            min_dist=cfg.umap_min_dist, n_components=cfg.umap_n_components, spread=cfg.umap_spread)
        embed = reducer.fit_transform(embed)
        joblib.dump(reducer, reducer_fname)
    else:
        print("UMAP projecting data", embed.shape)
        reducer = joblib.load(reducer_fname)
        embed = reducer.transform(embed)  

    return embed, scaler, reducer


@hydra.main(version_base=None, config_path="../../configs", config_name="knn_graph")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    # fix all random seeds
    fix_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = 'embed_projection'
    exp_dir = './'
    exp_dir = pathlib.Path(cfg.work_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    task_name='Embed_project'
    logger_path = os.path.join(exp_dir,'embed_project.log')

    logger = init_logger(logger_path, rank=0)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    encoder: Encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger)
    logger.info("Built {}.".format(encoder.model_name))

    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities,return_meta=True)

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

    #Get/make directories
    embed_dir = os.path.join(cfg.embed_dir,cfg.dataset.dataset_name)
    indices_dir = os.path.join(cfg.out_dir,cfg.dataset.dataset_name)
    out_dir = os.path.join(indices_dir, choices["encoder"])
    
    if not os.path.isdir(indices_dir):
        os.makedirs(indices_dir, exist_ok = True)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok = True)

    #Structure embedding test data
    embed_full = None
    target_full = None
    for batch_idx, data in enumerate(test_loader):
        if "filename" in data:
            image, target, image_fname, meta = data["image"], data["target"],data["filename"], data['metadata']
            image_fname = image_fname[0]
        else:
            image, target, meta = data["image"], data["target"], data['metadata']
            image_fname = meta['image_filename'][0]

        #Load embeddings for single input file
        embed_fname = os.path.join(embed_dir, choices["encoder"], "test", "embd_" + os.path.splitext(image_fname)[0] + ".npy")
        embed = np.load(embed_fname)

        print(embed_fname)
         
        #Rescale embedding to original dimension
        embed = rescale_embed(embed, cfg.dataset.img_size, device)
     

        #Flatten dimensions, except feature/channel dim

        target = target.flatten()

        #Get subset and apply indices, sampling each class available - subsetting done due to compuational complexity of tasks

        indices = get_indices(cfg, image_fname, target, indices_dir)
        
       
        sub_embed = embed[indices,:]
        sub_target = target[indices]



        #Merge individual subsets together
        if embed_full is None:
            embed_full = sub_embed
            target_full = sub_target
        else:
            embed_full = np.concatenate((sub_embed, embed_full))
            target_full = np.concatenate((sub_target, target_full))


    projection_data, scaler, reducer = train_and_gen_projection(embed_full, out_dir, cfg)

    del embed_full

    #Shift indices to start w/ zero - we can then use GeoTiff files for output / viz
    shift_1 = abs(min(projection_data[:,0]))
    shift_2 = abs(min(projection_data[:,1]))

    projection_data[:,0] = projection_data[:,0] + shift_1
    projection_data[:,1] = projection_data[:,1] + shift_2

    #Scale data to expand for viz.
    projection_data = (projection_data*10).astype(np.int32)

    max_ind_1 = int(max(projection_data[:,0]))
    max_ind_2 = int(max(projection_data[:,1]))
    final_projection = np.zeros((max_ind_1+1, max_ind_2+1), dtype=np.int32) - 1.0

    for i in range(target_full.shape[0]):
        final_projection[int(projection_data[i,0]), int(projection_data[i,1])] = target_full[i]

    ras_meta = {'driver': 'GTiff', 'dtype': 'int32', 'nodata': -1, 'width': final_projection.shape[1], 'height': final_projection.shape[0], 'count': 1, 'tiled': False, 'interleave': 'band'}

    out_file = os.path.join(out_dir, choices["encoder"] + ".UMAP_Labels.tif")
    with rasterio.open(out_file, 'w', **ras_meta) as dst:
        dst.write(final_projection, 1)

    print("Building KNN Graph")
    build_knn_graph(projection_data[:100], os.path.join(out_dir, choices["encoder"] + ".UMAP.knn_graph.zarr"))

   
if __name__ == "__main__":
    main()
