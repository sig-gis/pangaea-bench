# Adapted from https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/uper_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from pangaea.decoders.base import Decoder
from pangaea.decoders.ltae import LTAE2d, LTAEChannelAdaptor
from pangaea.encoders.base import Encoder

from timm.layers.conv_bn_act import ConvBnAct


class SegFormerHead(Decoder):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """
    def __init__(
            self,
            encoder: Encoder,
            finetune:str,
            channels:int,
            num_classes:int,
            act_layer = nn.ReLU, 
            norm_layer = nn.BatchNorm2d,
            pyramid_strategy='head_only',
            interpolate_mode='bilinear', 
            align_corners=False,
            in_channels: list[int] | None = None,
            feature_multiplier=1,
            **kwargs
        ):
        super().__init__(
            encoder=encoder,
            finetune=finetune,
            num_classes=num_classes
        )

        self.model_name = "SegFormer"
        self.encoder = encoder
        self.finetune = finetune
        self.num_classes = num_classes

        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.in_channels = in_channels
        self.channels = channels
        self.pyramid_strategy = pyramid_strategy
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners
        self.act_layer = act_layer
        self.feature_multiplier=1
    

        self.input_layers = self.encoder.output_layers
        self.input_layers_num = len(self.input_layers)

        if not self.finetune or self.finetune == 'none':
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif self.finetune == 'retrain_input':
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            self.encoder.unfreeze_input_layer()
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

        self.neck = Feature2Pyramid(
            embed_dim=self.in_channels,
            rescales=rescales,
        )

        self.convs = nn.ModuleList()
        for i in range(self.input_layers_num):

            self.convs.append(
                ConvBnAct(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer
                )
            )

        self.fusion_conv = ConvBnAct(
                    in_channels=self.channels*self.input_layers_num,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer
                )
        

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    

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

        outs = []

        if self.encoder.multi_temporal:
            if not self.finetune or self.finetune=='none':
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
            if not self.finetune or self.finetune=='none':
                with torch.no_grad():
                    feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})
            else:
                feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})
        
        if self.pyramid_strategy == 'head_only':
            feat = [feat[-1] for i in range(len(feat))]

        pyr = self.neck(feat)
        for idx in range(len(pyr)):
            x = pyr[idx]
            # print(x.shape)
            conv = self.convs[idx]
            outs.append(
                F.interpolate(
                    input=conv(x),
                    size=pyr[0].shape[2:], 
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            )
        # print(torch.cat(outs,dim=1).shape)
        out = self.fusion_conv(torch.cat(outs, dim=1))

        if not self.training:
            out = self.dropout(out)
        out = self.conv_seg(out)

        # fixed bug just for optical single modality
        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # interpolate to the target spatial dims
        output = F.interpolate(out, size=output_shape, mode="bilinear")

        return output



class SegFormerMTHead(SegFormerHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """
    def __init__(
            self,
            encoder: Encoder,
            finetune:str,
            channels:int,
            num_classes:int,
            multi_temporal:int,
            act_layer = nn.ReLU, 
            norm_layer = nn.BatchNorm2d,
            pyramid_strategy='head_only',
            interpolate_mode='bilinear', 
            align_corners=False,
            in_channels: list[int] | None = None,
            feature_multiplier=1,
            multi_temporal_strategy='ltae',
            **kwargs
        ):
        super().__init__(
            encoder=encoder,
            finetune=finetune,
            channels=channels,
            num_classes=num_classes,
            act_layer=act_layer,
            norm_layer=norm_layer,
            pyramid_strategy=pyramid_strategy,
            interpolate_mode=interpolate_mode,
            align_corners=align_corners,
            feature_multiplier=feature_multiplier,
            in_channels=in_channels,
        )

        self.model_name = "SegFormerMTHead" 
        self.multi_temporal = multi_temporal
        self.multi_temporal_strategy = multi_temporal_strategy

        decoder_in_channels = self.get_decoder_in_channels(self.multi_temporal_strategy,self.encoder)
        # if the encoder deals with multi_temporal inputs and
        # returns time merged outputs then we don't need multi_temporal_strategy
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

        outs = []

        # If the encoder handles multi_temporal we feed it with the input
        if self.encoder.multi_temporal:
            if not self.finetune or self.finetune=='none':
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
                if not self.finetune or self.finetune=='none':
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
        
        if self.pyramid_strategy == 'head_only':
            feat = [feats[-1] for i in range(len(feats))]

        pyr = self.neck(feat)
        for idx in range(len(pyr)):
            x = pyr[idx]
            # print(x.shape)
            conv = self.convs[idx]
            outs.append(
                F.interpolate(
                    input=conv(x),
                    size=pyr[0].shape[2:], 
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            )
        # print(torch.cat(outs,dim=1).shape)
        out = self.fusion_conv(torch.cat(outs, dim=1))

        if not self.training:
            out = self.dropout(out)
        out = self.conv_seg(out)

        # fixed bug just for optical single modality
        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # interpolate to the target spatial dims
        output = F.interpolate(out, size=output_shape, mode="bilinear")

        return output    



    # def forward(self, inputs):
    #     # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
    #     inputs = self._transform_inputs(inputs)
    #     outs = []
    #     for idx in range(len(inputs)):
    #         x = inputs[idx]
    #         conv = self.convs[idx]
    #         outs.append(
    #             resize(
    #                 input=conv(x),
    #                 size=inputs[0].shape[2:],
    #                 mode=self.interpolate_mode,
    #                 align_corners=self.align_corners))

    #     out = self.fusion_conv(torch.cat(outs, dim=1))

    #     out = self.cls_seg(out)

    #     return out




class Feature2Pyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
    """

    def __init__(
        self,
        embed_dim,
        rescales=(4, 2, 1, 0.5),
    ):
        super().__init__()
        self.rescales = rescales
        self.upsample_4x = None
        self.ops = nn.ModuleList()

        for i, k in enumerate(self.rescales):
            if k == 4:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                        nn.SyncBatchNorm(embed_dim[i]),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        ),
                    )
                )
            elif k == 2:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i], kernel_size=2, stride=2
                        )
                    )
                )
            elif k == 1:
                self.ops.append(nn.Identity())
            elif k == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif k == 0.25:
                self.ops.append(nn.MaxPool2d(kernel_size=4, stride=4))
            else:
                raise KeyError(f"invalid {k} for feature2pyramid")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []

        for i in range(len(inputs)):
            outputs.append(self.ops[i](inputs[i]))
        return tuple(outputs)

# adapted from: https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/utils/wrappers.py#L8
# def resize(input,
#            size=None,
#            scale_factor=None,
#            mode='nearest',
#            align_corners=None
#         ):
#         if size is not None and align_corners:
#             input_h, input_w = tuple(int(x) for x in input.shape[2:])
#             output_h, output_w = tuple(int(x) for x in size)
#             if output_h > input_h or output_w > output_h:
#                 if ((output_h > 1 and output_w > 1 and input_h > 1
#                      and input_w > 1) and (output_h - 1) % (input_h - 1)
#                         and (output_w - 1) % (input_w - 1)):
#                     # warnings.warn(
#                     #     f'When align_corners={align_corners}, '
#                     #     'the output would more aligned if '
#                     #     f'input size {(input_h, input_w)} is `x+1` and '
#                     #     f'out size {(output_h, output_w)} is `nx+1`')
#                     return F.interpolate(input, size, scale_factor, mode, align_corners)