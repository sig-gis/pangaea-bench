import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import hashlib
import os as os
import pathlib
import pprint
import time

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pangaea.utils.logger import init_logger
from pangaea.utils.subset_sampler import get_subset_indices
from pangaea.utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_final_model_ckpt_path,
    get_generator,
    seed_worker,
)


import scipy
from scipy.spatial.distance import pdist, squareform

#- Graph-stats 
from graspologic.embed import OmnibusEmbed, ClassicalMDS, AdjacencySpectralEmbed
from graspologic.simulations import rdpg

from tqdm import tqdm
import os
import numpy as np
import zarr



def bootstrap_null(graph, number_of_bootstraps=25, n_components=None, umap_n_neighbors=32, acorn=None, fname_uid=""):
    '''
    Constructs a bootstrap null distribution for the difference of latent positions of the nodes in the passed graph
    :param graph: [N, N] binary symmetric hollow matrix to model
    :param number_of_bootstraps: the number of bootstrap replications
    :param n_components: the number of components to use in initial ASE. selected automatically if None.
    :param umap_n_neighbors: the number of neighbors to use in umap
    :param acorn: rng seed to control for randomness in umap and ase
    :return: [2, N, number_of_bootstraps], n_components.
             The 0 column of the matrix is the ASE estimates, and the 1 column is the UMAP estimates.
             n_components is the number of selected components
    '''
    if acorn is not None:
        np.random.seed(acorn)

    ase_latents = AdjacencySpectralEmbed(n_components=n_components, svd_seed=acorn).fit_transform(graph)

    n, n_components = ase_latents.shape

    distances = np.zeros((number_of_bootstraps, n))

    for i in tqdm(range(number_of_bootstraps)):
        graph_b = rdpg(ase_latents, directed=False)

        bootstrap_latents = OmnibusEmbed(n_components=n_components).fit_transform([graph, graph_b])
        distances[i] = np.linalg.norm(bootstrap_latents[0] - bootstrap_latents[1], axis=1)
    return distances.transpose((1, 0)) , n_components


def get_cdf(pvalues, num=26):
    linspace = np.linspace(0, 1, num=num)

    cdf = np.zeros(num)

    for i, ii in enumerate(linspace):
        cdf[i] = np.mean(pvalues <= ii)

    return cdf



def build_dist_mtx(model_names, knn_graphs):
    dist_matrix = np.zeros((len(model_names), len(model_names)))
    for i, embed_function1 in enumerate(model_names):
        for j, embed_function2 in enumerate(model_names[i+1:], i+1):
            omni_embds = OmnibusEmbed(n_components=2).fit_transform([knn_graphs[embed_function1], knn_graphs[embed_function2]])
            temp_dist = np.linalg.norm(omni_embds[0] - omni_embds[1]) / np.linalg.norm( (omni_embds[0] + omni_embds[1])) # / 2 )
            dist_matrix[i,j] = temp_dist
            dist_matrix[j,i] = temp_dist
    return dist_matrix

def run_nomic_analysis(model_names, knn_graphs, out_dir):

    #- now, we can "easily" learn a joint/aligned low-dimensional embedding of the two sets of embeddings
    omni_embds = OmnibusEmbed(n_components=2).fit_transform(list(knn_graphs.values())) #list()

    colors = ['red', 'black', 'blue', 'orange', 'green', "magenta", "cyan", "olive", "purple", \
              "gray", "pink", "brown", "darkcyan", "chocolate", "lightgreen", "gold", "deeppink",\
              "lightgrey", "rosybrown", "maroon", "coral", "sandybrown"]
    fig, ax = plt.subplots()
    for i, model_name in enumerate(model_names): # < 5 < 40 < 5
        ax.scatter(omni_embds[i,:,0], omni_embds[i, :,1], c=colors[i], label=model_name)
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    plt.show()
    print("Plotting Scatter Embed")
    plt.savefig(os.path.join(out_dir, "Embed_Scatter_multi" + ".png"))
    plt.clf()
    plt.close(fig)    

    for i, model_name in enumerate(model_names): # < 5 < 40 < 5
        fig, ax = plt.subplots()
        ax.scatter(omni_embds[i,:,0], omni_embds[i, :,1], c=colors[i], label=model_name, alpha = 0.2)
        plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


        plt.show()
        print("Plotting Scatter Embed")
        plt.savefig(os.path.join(out_dir, "Embed_Scatter_" + model_names[i] + ".png"))
        plt.clf()
        plt.close(fig)
    


    #- A simple "check" to see if the two embedding functions represent sample i differently
    #- is to look at the distance || omni[0][i] - omni[1][i] ||
    argsorted=np.argsort(np.linalg.norm(omni_embds[0] - omni_embds[1], axis=1))
 
    #- i.e., dataset[argsorted[0]] has moved the most

    #- A more statistically rigorous way to determine *if* a sample has moved
    #- is to use the hypothesis test described in the paper

    for i, model_name in enumerate(model_names):
        for j in range(0,len(model_names)):
            #if i == j:
            #    continue
            model_name2 = model_names[j]
            print(model_name2, model_name)
            null_dist, ase_n_components  = bootstrap_null(knn_graphs[model_names[i]], n_components=2, \
                number_of_bootstraps=100, fname_uid="_" + model_names[i] + "_" + model_names[j])
            test_statistics = np.linalg.norm(omni_embds[i] - omni_embds[j], axis=1)
            p_values = []


            for st, test_statistic in enumerate(test_statistics):
                p_value = np.mean(test_statistic <= null_dist[st])
                p_values.append(p_value)
    
    
            #- same joint embedding space as above, but this time just plotting nomic-ai
            #- and adding color intensity to represent p-value
            fig, ax = plt.subplots()
            for d in range(omni_embds.shape[1]):
                ax.scatter(omni_embds[j, d, 0], omni_embds[j, d, 1], label=model_name, color=colors[j])
                ax.scatter(omni_embds[i, d, 0], omni_embds[i, d, 1], label=model_name, color=colors[i], alpha=1-p_values[d])
            plt.show()
            print("Plotting Null Hyp Scatter")
            plt.savefig(os.path.join(out_dir, "Embed_Scatter_Null_Hyp_" + model_name + "_" + model_name2 + ".png"))
            plt.clf()
            plt.close(fig) 
            #- Notice that the ranking of the p-values is related to but does not equal ranking of || omni[0][i] - omni[1][i] ||

            #- Looking at distribution of p-values relative to the uniform dist
            #- there doesnt seem to be a systematic difference

            linspace=np.linspace(0, 1, num=25+1)
            cdf  = get_cdf(p_values, num=25+1)

            fig, ax = plt.subplots(1,1)

            ax.plot(linspace, cdf, label='observed')
            ax.plot(linspace, linspace, label='null / uniform (y=x)')
            ax.legend()
            plt.show()
            print("Plotting P-Value")
            plt.savefig(os.path.join(out_dir, "P_Value_Dist_" + model_name + "_" + model_name2 + ".png"))
            plt.clf()
            plt.close(fig)
    fig, ax = plt.subplots()
    #- Get low-dimensional representations of embedding models.
    #- "Families" of embedding models are close to each other in this space.
    dist_matrix = build_dist_mtx(model_names, knn_graphs)
 

    colors = ['red', 'black', 'blue', 'orange', 'green', "magenta", "cyan", "olive", "purple", \
              "gray", "pink", "brown", "darkcyan", "chocolate", "lightgreen", "gold", "deeppink",\
              "lightgrey", "rosybrown", "maroon", "coral", "sandybrown"]
 
    cmds_embds = ClassicalMDS(n_components=2).fit_transform(dist_matrix)
    for i, cmds in enumerate(cmds_embds):
        ax.scatter(cmds[0], cmds[1], label=model_names[i], c=colors[i])

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
 

    plt.show()
    plt.savefig(os.path.join(out_dir, "Embed_Space_Dist_Mtx.png"))
    plt.close(fig)

@hydra.main(version_base=None, config_path="../../configs", config_name="embed_compare")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = 'embed_compare'
    exp_dir = './'
    exp_dir = pathlib.Path(cfg.work_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    task_name='Embed_compare'
    logger_path = os.path.join(exp_dir,'embed_compare.log')

    logger = init_logger(logger_path, rank=0)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    out_dir = cfg.out_dir


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

     


    if int(cfg.dataset.multi_temporal) > 0:
        model_names = [
            "croma_optical",
            "dofa",
            "prithvi",
            "satlasnet_mi",
            "scalemae",
            "spectralgpt",
            "ssl4eo_data2vec",
            "ssl4eo_dino",
            "ssl4eo_mae_optical",
            "ssl4eo_moco",
            "unet_encoder_mi",
            "terramind_large",
            "vit_scratch",
             "vit_mi"
        ]
        # model_names = [
        #"croma_optical",
        #"dofa",
        #"prithvi"]


    choices = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    out_dir = os.path.join(cfg.out_dir,cfg.dataset.dataset_name)

    knn_graphs = {}
    mx_dim_1 = 0
    mx_dim_0 = 0
    for key in model_names:
        print(os.path.join(out_dir, key, key + ".UMAP.knn_graph.zarr"))
        knn_graphs[key] = zarr.load(os.path.join(out_dir, key, key + ".UMAP.knn_graph.zarr")).astype(np.float32)
        mx_dim_1 = max(mx_dim_1, knn_graphs[key].shape[1])
        mx_dim_0 = max(mx_dim_0, knn_graphs[key].shape[0])
        #knn_graphs[key] = knn_graphs[key].astype(np.int8)

    #for key in model_names:
    #    new_arr = np.zeros((mx_dim_0, mx_dim_1), dtype=np.int8)
    #    new_arr[0:knn_graphs[key].shape[0], 0:knn_graphs[key].shape[1]] = knn_graphs[key]
    #    knn_graphs[key] = new_arr
    #    print(new_arr.shape)

    run_nomic_analysis(model_names, knn_graphs, out_dir)

				                


if __name__ == "__main__":
    main()


