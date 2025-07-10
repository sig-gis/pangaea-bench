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

from pangaea.decoders.utils import ConvModule,Feature2Pyramid,PPM,build_act,resize,build_norm_layer

from einops import rearrange

class SETRMLAHead(Decoder):

    def __init__(
            self,
            encoder,
            num_classes,
            finetune,
            mla_channels=128,
            upscale=4,
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'),
            align_corners = False,
            in_channels:list[int] | None = None

    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.mla_channels = mla_channels
        self.upscale = upscale
        self.in_channels = in_channels
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners

        num_inputs = len(self.in_channels)

        self.up_convs = nn.ModuleList()
        for i in range(num_inputs):
            self.up_convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=mla_channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    ConvModule(
                        in_channels=mla_channels,
                        out_channels=mla_channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    Upsample(
                        scale_factor=upscale,
                        mode='bilinear',
                        align_corners=self.align_corners
                    )
                )
            )

        self.conv_seg = nn.Conv2d(in_channels=self.num_inputs*self.mla_channels,out_channels=self.num_classes,kernel_size=1)
    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """Compute the segmentation output.

        Args:
            img (dict[str, torch.Tensor]): input data structured as a dictionary:
            img = {modality1: tensor1, modality2: tensor2, ...}, e.g. img = {"optical": tensor1, "sar": tensor2}.
            with tensor1 and tensor2 of shape (B C T=1 H W) with C the number of encoders'bands for the given modality.
            output_shape (torch.Size | None, optional): output's spatial dims (H, W) (equals to the target spatial dims).
            Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (B, num_classes, H', W') with (H' W') coressponding to the output_shape.
        """

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

        outs = []

        for x, up_conv in zip(feat,self.up_convs):
            outs.append(up_conv(x))
        out = torch.cat(outs,dim=1)
        out = self.conv_seg(out)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # interpolate to the target spatial dims
        output = F.interpolate(out, size=output_shape, mode="bilinear")

        return output

class SETRUPHead(Decoder):

    def __init__(
            self,
            encoder,
            num_classes,
            finetune,
            channels=768,
            num_convs=1,
            kernel_size=3,
            upscale=4,
            init_cfg = [
                dict(type='Constant',val=1.0,bias=0,layer='LayerNorm'),
                dict(type='Normal',std=0.01,ocerride=dict(name='conv_seg'))
            ],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='LN',eps=1e-6),
            align_corners = False,
            in_channels:list[int] | None = None

    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.channels = channels
        self.upscale = upscale
        self.in_channels = in_channels
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.init_cfg = init_cfg
        self.num_convs = num_convs
        self.kernel_size=kernel_size

        self.norm = build_norm_layer(norm_cfg=self.norm_cfg,num_features=self.in_channels)

        self.up_convs = nn.ModuleList()
        in_channels = self.in_channels
        out_channels = self.channels

        for _ in range(num_convs):
            self.up_convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(kernel_size-1) // 2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    Upsample(
                        scale_factor=upscale,
                        mode='bilinear',
                        align_corners=self.align_corners
                    )
                )
            )
            in_channels=out_channels
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes)
    
    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """Compute the segmentation output.

        Args:
            img (dict[str, torch.Tensor]): input data structured as a dictionary:
            img = {modality1: tensor1, modality2: tensor2, ...}, e.g. img = {"optical": tensor1, "sar": tensor2}.
            with tensor1 and tensor2 of shape (B C T=1 H W) with C the number of encoders'bands for the given modality.
            output_shape (torch.Size | None, optional): output's spatial dims (H, W) (equals to the target spatial dims).
            Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (B, num_classes, H', W') with (H' W') coressponding to the output_shape.
        """

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
        n,c,h,w = feat.shape
        x = x.reshape(n,c,h*w).transpose(2,1).contiguous()
        x = self.norm(x)
        x = x.transpose(1,2).reshape(n,c,h,w).contiguous()

        for up_conv in self.up_convs:
            x = up_conv(x)
        
        out = self.conv_seg(x)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # interpolate to the target spatial dims
        output = F.interpolate(out, size=output_shape, mode="bilinear")

        return output
    


        



class Upsample(nn.Module):
    def __init__(
            self,
            size=None,
            scale_factor=None,
            mode='nearest',
            align_corners=None
        ):
        super().__init__()

        self.size=size

        if isinstance(scale_factor,tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self,x):
        if not self.size:
            size = [int(t*self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size

        return resize(x,size,None,self.mode,self.align_corners)
        
