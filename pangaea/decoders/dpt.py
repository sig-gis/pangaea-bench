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

class DPTHead(Decoder):

    def __init__(self,
                 encoder,
                 num_classes,
                 finetune,
                 embed_dims=768,
                 channels=512,
                 readout_type='ignore',
                 post_process_channels=[96,192,384,768],
                 patch_size=16,
                 expand_channels=False,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 in_channels:list[int] | None = None
            ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune
        )

        self.channels=channels
        self.readout_type=readout_type
        self.post_process_channels = post_process_channels
        self.patch_size = patch_size
        self.expand_channels = expand_channels
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.in_channels = in_channels

        self.in_channels = self.channels
        
        self.reassemble_blocks = ReassembleBlocks(
            embed_dims,
            post_process_channels,
            readout_type,
            patch_size
        )

        self.post_process_channels = [
            channel*math.pow(2,i) if expand_channels else channel for i,channel in enumerate(post_process_channels)
        ]
        self.convs = nn.ModuleList()

        for channel in self.post_process_channels:
            self.convs.append(
                ConvModule(
                    channel,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    act_cfg=None,
                    bias=False
                )
            )
        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(FeatureFusionBlock(self.channels,act_cfg,norm_cfg))
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg
        )
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)

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

        x = self.reassemble_blocks(feat)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1,len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out,x[-(i+1)])
        out = self.project(out)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # interpolate to the target spatial dims
        output = F.interpolate(out, size=output_shape, mode="bilinear")

        return output



class ReassembleBlocks(nn.Module):
    def __init__(self,
                 in_channels=768,
                 out_channels = [96,192,384,768],
                 readout_type='ignore',
                 patch_size=16,
                 init_cfg=None
            ):
        self.readout_type = readout_type
        self.patch_size=patch_size

        self.projects = nn.ModuleList([
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                act_cfg=None
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0
            ),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0
            ),
            nn.Identity(),
            nn.Conv2d(
                in_channels=in_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            )
        ])

        if self.readout_type == 'project':
            self.readout_projects = nn.ModuleList()

            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2*in_channels,in_channels),
                        build_act(dict(type='GELU'))
                    )
                )
    def forward(self,inputs):
        out = []
        for i,x in enumerate(inputs):
            x, cls_token  = x[0],x[1]
            feature_shape = x.shape

            if self.readout_type == 'project':
                x = x.flatten(2).permute((0,2,1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x,readout),-1))
            elif self.readout_type == 'add':
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        return out
    
class PreActResidualConvUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 act_cfg,
                 norm_cfg,
                 stride=1,
                 dilation=1,
                 init_cfg=None
                ):
        super().__init__()

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=('act','conv','norm')
        )

        self.conv2 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=('act','conv','norm')
        )

    def forward(self,inputs):
        inputs_ = inputs.clone()
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs_
    
class FeatureFusionBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            act_cfg,
            norm_cfg,
            expand=False,
            align_corners=False,
            init_cfg=None     
        ):
        super().__init__()

        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners

        self.out_channels = in_channels

        if self.expand:
            self.out_channels = in_channels // 2

        self.project = ConvModule(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            act_cfg=None,
            bias=True
        )

        self.res_conv_unit1 = PreActResidualConvUnit(
            in_channels=self.in_channels,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.res_conv_unit2 = PreActResidualConvUnit(
            in_channels=self.in_channels,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
    def forward(self,*inputs):
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = resize(
                    inputs[1],
                    size=(x.shape[2],x.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = resize(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=self.align_corners
        )
        x = self.project(x)

        return x




