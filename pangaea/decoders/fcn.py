import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import PixelShuffle

from copy import deepcopy

from timm.models.layers import to_2tuple
from timm.layers import create_norm,create_act

from pangaea.decoders.base import Decoder
from pangaea.decoders.ltae import LTAE2d, LTAEChannelAdaptor
from pangaea.encoders.base import Encoder

from pangaea.decoders.utils import ConvModule,Feature2Pyramid,PPM

from einops import rearrange

class FCNDecoder(Decoder):
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: str,
        channels: int,
        num_convs=2,
        kernel_size=3,
        concat_input=True,
        dilation=1,
        interp_method = 'interp',
        pool_scales=(1, 2, 3, 6),
        feature_multiplier: int = 1,
        in_channels: list[int] | None = None,  
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN'),
        conv_cfg=dict(type='Conv2d'),
        init_cfg=None
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune
        )

        self.model_name = 'FCN'
        self.encoder = encoder
        self.finetune = finetune
        self.channels = channels
        self.num_convs = num_convs
        self.kernel_size = kernel_size
        self.concat_input = concat_input
        self.dilation = dilation
        self.interp_method = interp_method
        self.pool_scales = pool_scales
        self.feature_multiplier = feature_multiplier
        self.in_channels = in_channels
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.init_cfg = init_cfg

        if not self.finetune or self.finetune == 'none':
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif self.finetune == 'retrain_input':
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            self.encoder.unfreeze_input_layer()

        self.input_layers = self.encoder.output_layers
        self.input_layers_num = len(self.input_layers)

        if in_channels is None:
            self.in_channels = [
                dim * feature_multiplier for dim in self.encoder.output_dim
            ]
        else:
            self.in_channels = [dim * feature_multiplier for dim in in_channels]

        if self.encoder.pyramid_output:
            rescales = [1 for _ in range(self.input_layers_num)]
        else:
            scales = [4, 2, 1, 0.5]
            rescales = [
                scales[int(i / self.input_layers_num * 4)]
                for i in range(self.input_layers_num)
            ]


        conv_padding = (kernel_size // 2) * dilation

        self.in_channels = self. in_channels[-1]    

        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=self.dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        )
        for i in range(num_convs-1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=self.dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            ) 
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )

        self.dropout = nn.Dropout2d(0.1)

        if interp_method == 'PixelShuffle':
            (H,W,C) = self.encoder.output_shape
            
            r = self.encoder.input_size // H 
            self.conv_seg = ConvModule(
                in_channels=self.channels,
                out_channels=self.num_classes * r * r,
                kernel_size=1
            )
            self.pixel_shuffle = PixelShuffle(upscale_factor=r)
        elif interp_method == 'interpolate':
            self.conv_seg = ConvModule(
                in_channels=self.channels,
                out_channels=self.num_classes,
                kernel_size=1
            )

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
         # img[modality] of shape [B C T=1 H W]
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder(img)
            else:
                feat = self.encoder(img)

            # multi_temporal models can return either (B C' T=1 H' W')
            # or (B C' H' W'), we need (B C' H' W')
            if self.encoder.multi_temporal_output:
                feat = [f.squeeze(-3) for f in feat]

        else:
            # remove the temporal dim
            # [B C T=1 H W] -> [B C H W]
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})
            else:
                feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})

        feat = feat[-1]

        x = self.convs(feat)

        if self.concat_input:
            x = self.conv_cat(torch.cat([x,feat],dim=1))

        x = self.conv_seg(x)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]
        
        if self.interp_method == 'interpolate':
            output = F.interpolate(x,size=output_shape,mode='bilinear')
        elif self.interp_method == 'PixelShuffle':
            output = self.pixel_shuffle(x)

        return output