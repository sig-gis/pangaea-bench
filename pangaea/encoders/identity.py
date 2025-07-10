from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Block

from einops import rearrange

from pangaea.encoders.base import Encoder
from pangaea.encoders.pos_embed import get_3d_sincos_pos_embed

class IdentityEncoder(Encoder):
    def __init__(self,
        input_bands: dict[str, list[str]],
        input_size: int,
    ):
        
        output_dim = len(input_bands['optical'])
        if 'sar' in input_bands:
            output_dim = output_dim + len(input_bands['sar'])
        super().__init__(
            model_name='Identity',
            input_bands=input_bands,
            input_size=input_size,
            output_layers=None,
            output_dim=output_dim,
            multi_temporal=True,
            multi_temporal_output=True,
            pyramid_output=False,
        )

    def load_encoder_weights(self, logger):
        pass

    def forward(self, image):

        x = image['optical']
        if 'sar' in image:
            sar = image['sar']

            x = torch.cat([x,sar],dim=1)
        
        return x

def patchify(imgs,patch_size):
    """
    imgs: B, C, H, W
    x: B, L, D
    """
    p = patch_size
    x = rearrange(imgs, 'b c (h p) (w q) -> b (h w) (p q c)', p=p, q=p)

    return x

def unpatchify(x,patch_size,img_size):
    """
    x: B, L, D
    imgs: B, C, H, W
    """
    p = patch_size
    num_p = img_size // p
    imgs = rearrange(x, 'b (h w) (p q c) -> b c (h p) (w q)', h=num_p, w=num_p, p=p, q=p)
    return imgs