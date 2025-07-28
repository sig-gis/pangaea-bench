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

class FCNDecoderMT(FCNDecoder):
    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: str,
        channels: int,
        multi_temporal:int,
        multi_temporal_strategy:str |None,
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
            encoder=self.encoder,
            num_classes = self.num_classes,
            finetune = self.finetune,
            channels = self.channels,
            num_convs = self.num_convs,
            kernel_size = self.kernel_size,
            concat_input = self.concat_input,
            dilation=self.dilation,
            interp_method=self.interp_method,
            pool_scales=self.pool_scales,
            feature_multiplier=self.feature_multiplier,
            in_channels=self.in_channels,
            act_cfg=self.act_cfg,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
            init_cfg=self.init_cfg
        )

        self.model_name = 'FCNMT'
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

        decoder_in_channels = self.get_decoder_in_channels(
            multi_temporal_strategy, encoder
        )

        self.multi_temporal = multi_temporal
        self.multi_temporal_strategy = multi_temporal_strategy

        if self.encoder.multi_temporal and not self.encoder.multi_temporal_output:
            self.tmap = None

        else:
            if self.multi_temporal_strategy == "ltae":
                ltae_in_channels = max(decoder_in_channels)
                # if the encoder output channels vary we must use an adaptor before the LTAE
                if decoder_in_channels != encoder.output_dim:
                    self.ltae_adaptor = LTAEChannelAdaptor(
                        in_channels=encoder.output_dim,
                        out_channels=decoder_in_channels,
                    )
                else:
                    self.ltae_adaptor = lambda x: x
                self.tmap = LTAE2d(
                    positional_encoding=False,
                    in_channels=ltae_in_channels,
                    mlp=[ltae_in_channels, ltae_in_channels],
                    d_model=ltae_in_channels,
                )
            elif self.multi_temporal_strategy == "linear":
                self.tmap = nn.Linear(self.multi_temporal, 1)
            else:
                self.tmap = None

    def get_decoder_in_channels(
        self, multi_temporal_strategy: str | None, encoder: Encoder
    ) -> list[int]:
        if multi_temporal_strategy == "ltae":
            # if the encoder output channels vary we must use an adaptor before the LTAE
            ltae_in_channels = max(encoder.output_dim)
            if ltae_in_channels != min(encoder.output_dim):
                return [ltae_in_channels for _ in encoder.output_dim]
        return encoder.output_dim

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """Compute the segmentation output for multi-temporal data.

        Args:
            img (dict[str, torch.Tensor]): input data structured as a dictionary:
            img = {modality1: tensor1, modality2: tensor2, ...}, e.g. img = {"optical": tensor1, "sar": tensor2}.
            with tensor1 and tensor2 of shape (B C T H W) with C the number of encoders'bands for the given modality,
            and T the number of time steps.
            output_shape (torch.Size | None, optional): output's spatial dims (H, W) (equals to the target spatial dims).
            Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (B, num_classes, H', W') with (H' W') coressponding to the output_shape.
        """
        # If the encoder handles multi_temporal we feed it with the input
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feats = self.encoder(img)
            else:
                feats = self.encoder(img)
            # multi_temporal models can return either (B C' T H' W')
            # or (B C' H' W') via internal merging strategy

        # If the encoder handles only single temporal data, we apply multi_temporal_strategy
        else:
            feats = []
            for i in range(self.multi_temporal):
                if not self.finetune:
                    with torch.no_grad():
                        feats.append(
                            self.encoder({k: v[:, :, i, :, :] for k, v in img.items()})
                        )
                else:
                    feats.append(
                        self.encoder({k: v[:, :, i, :, :] for k, v in img.items()})
                    )

            feats = [list(i) for i in zip(*feats)]
            # obtain features per layer
            feats = [torch.stack(feat_layers, dim=2) for feat_layers in feats]

        if self.tmap is not None:
            if self.multi_temporal_strategy == "ltae":
                feats = self.ltae_adaptor(feats)
                feats = [self.tmap(f) for f in feats]
            elif self.multi_temporal_strategy == "linear":
                feats = [self.tmap(f.permute(0, 1, 3, 4, 2)).squeeze(-1) for f in feats]

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