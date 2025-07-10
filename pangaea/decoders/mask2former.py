import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import PixelShuffle

import math
from copy import deepcopy

from timm.models.layers import to_2tuple
from timm.layers import create_norm,create_act

from pangaea.decoders.base import Decoder
from pangaea.decoders.ltae import LTAE2d, LTAEChannelAdaptor
from pangaea.encoders.base import Encoder

from pangaea.decoders.utils import ConvModule,Feature2Pyramid,PPM,build_act,resize

from einops import rearrange

class Mask2Former(Decoder):

    def __init__(
            self,
            encoder,
            num_classes,
            finetune
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune
        )