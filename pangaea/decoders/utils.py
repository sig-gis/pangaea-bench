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

class ConvModule(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias='auto',
                conv_cfg = dict(type='Conv2d'),
                norm_cfg= None,
                act_cfg = dict(type='ReLU'),
                inplace=True,
                with_spectral_norm=False,
                padding_mode='zeros',
                order=('conv','norm','act'),       
        ):

        super().__init__()
        official_padding_mode = ['zeros','circular']

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg)
        
        conv_padding = 0 if self.with_explicit_padding else padding

        conv_cfg['in_channels'] = in_channels
        conv_cfg['out_channels'] = out_channels
        conv_cfg['kernel_size'] = kernel_size
        conv_cfg['stride'] = stride
        conv_cfg['padding'] = conv_padding
        conv_cfg['dilation'] = dilation
        conv_cfg['groups'] = groups
        conv_cfg['bias'] = bias

        self.conv = build_conv_layer(conv_cfg=conv_cfg)
        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            
            norm = build_norm_layer(norm_cfg,norm_channels)
            self.norm = norm
            self.add_module('norm',norm)
        
        if self.with_activation:
            _act_cfg = act_cfg.copy()
            act = build_act(_act_cfg)

            self.activate = act
    def forward(self,x,activate=True,norm=True):
        layer_index = 0

        while layer_index < len(self.order):
            layer = self.order[layer_index]
            if layer == 'conv':
                if self.with_explicit_padding:
                        x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)

            layer_index += 1

        return x


class FFN(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU',inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 layer_scale_init_value=0.
            ):
        super(FFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims

        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels,feedforward_channels),
                    build_act(act_cfg),
                    nn.Dropout(ffn_drop)
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels,embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims,scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

    def forward(self,x,identity=None):
        out = self.layers(x)
        out = self.gamma2(out)

        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
    
class LayerScale(nn.Module):
    def __init__(self,
                 dim,
                 inplace=False,
                 data_format='channels_last',
                 scale=1e-5
            ):
        super().__init__()
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim)*scale)

    def forward(self,x):
        if self.data_format == 'channels_first':
            shape = tuple((1,-1,*(1 for _ in range(x.dim() - 2))))
        else:
            shape = tuple((*(1 for _ in range(x.dim() - 1)),-1))
        
        if self.inplace:
            return x.mul_(self.weight.view(*shape))
        else:
            return x*self.weight.view(*shape)

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.channels,
                        kernel_size=1,
                        padding=0,
                    ),
                    nn.SyncBatchNorm(self.channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

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
                            embed_dim[i], embed_dim[i] // 4, kernel_size=2, stride=2
                        ),
                    )
                )
            elif k == 2:
                self.ops.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim[i], embed_dim[i] // 2, kernel_size=2, stride=2
                        )
                    )
                )
            elif k == 1:
                self.ops.append(nn.Identity())
            elif k == 0.5:
                # self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
                self.ops.append(
                    nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(embed_dim[i],embed_dim[i]*2,kernel_size=1,stride=1)
                    )
                )
            elif k == 0.25:
                self.ops.append(nn.MaxPool2d(kernel_size=4, stride=4))
            elif k == 0.125:
                self.ops.append(nn.MaxPool2d(kernel_size=8, stride=8))
            else:
                raise KeyError(f"invalid {k} for feature2pyramid")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []

        for i in range(len(inputs)):
            outputs.append(self.ops[i](inputs[i]))
        return tuple(outputs)
    
def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
    # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
    
def build_norm_layer(
        norm_cfg=None,
        num_features=512
                    ):
    norm_cfg = deepcopy(norm_cfg)
    layer_type = norm_cfg.pop('type')
    if layer_type == "LN":
        norm = nn.LayerNorm(num_features,**norm_cfg)
    elif layer_type == 'BN':
        norm = nn.BatchNorm2d(num_features,**norm_cfg)
    elif layer_type =='SyncBN':
        norm = nn.SyncBatchNorm(num_features,**norm_cfg)

    return norm
def build_padding_layer(cfg):
    cfg = deepcopy(cfg)
    if cfg['type'] == 'zero':
        cfg.pop('type')
        layer = nn.ZeroPad2d(**cfg)
    elif cfg['type']=='reflect':
        cfg.pop('type')
        layer = nn.ReflectionPad2d(**cfg)
    elif cfg['type'] == 'replicate':
        cfg.pop('type')
        layer = nn.ReplicationPad2d(cfg)

    return layer

def build_act(act_cfg):
    act_cfg = deepcopy(act_cfg)
    act_type = act_cfg.pop('type')
    if act_type == 'ReLU':
        return nn.ReLU(**act_cfg)
    elif act_type == 'PReLU':
        return nn.PReLU(**act_cfg)
    elif act_type == 'GELU':
        return nn.GELU(**act_cfg)
    elif act_type == 'Sigmoid':
        return nn.Sigmoid()
    elif act_type == 'Tanh':
        return nn.Tanh()
    
def build_conv_layer(conv_cfg):
    conv_cfg = deepcopy(conv_cfg)
    conv_type = conv_cfg.pop('type')

    if conv_type == 'Conv2d' or conv_type == 'Conv':
        return nn.Conv2d(**conv_cfg)
    
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None
        ):
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    # warnings.warn(
                    #     f'When align_corners={align_corners}, '
                    #     'the output would more aligned if '
                    #     f'input size {(input_h, input_w)} is `x+1` and '
                    #     f'out size {(output_h, output_w)} is `nx+1`')
                    return F.interpolate(input, size, scale_factor, mode, align_corners)