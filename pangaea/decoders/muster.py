# Adapted from https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/decode_heads/uper_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import PixelShuffle

from copy import deepcopy

from timm.models.layers import to_2tuple
from timm.layers import create_norm,create_act

from pangaea.decoders.utils import resize

from pangaea.decoders.base import Decoder
from pangaea.decoders.ltae import LTAE2d, LTAEChannelAdaptor
from pangaea.encoders.base import Encoder

from einops import rearrange

class MusterDecoder(Decoder):
    def __init__(
            self,
            encoder: Encoder,
            num_classes:int,
            finetune:str,
            # embed_dims:int,
            channels:int,
            patch_size:int,
            window_size:int,
            mlp_ratio:int,
            depths:list[int],
            num_heads:list[int],
            strides:list[int],
            qkv_bias:bool,
            qk_scale:float,
            drop_rate:float,
            attn_drop_rate:float,
            drop_path_rate:float,
            act_cfg:dict,
            in_index:list[int],
            norm_cfg:dict,
            align_corners:bool,
            in_channels:list[int] | None = None,
            pool_scales= (1,2,3,6),
            pyramid_strategy='head',
            interp_method = 'PixelShuffle',
            feature_multiplier:int=1
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.model_name = "MUSTER"
        self.encoder = encoder
        self.finetune = finetune
        self.feature_multiplier = feature_multiplier

        self.patch_size = patch_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.num_heads = num_heads
        self.strides = strides
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners
        self.pool_scales = pool_scales
        self.pyramid_strategy = pyramid_strategy
        self.interp_method = interp_method

        if not self.finetune or self.finetune == 'none':
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif self.finetune == 'retrain_input':
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            self.encoder.unfreeze_input_layer()

        self.input_layers = self.encoder.output_layers
        self.input_layers_num = len(self.input_layers)

        H, W, C = self.encoder.output_shape
        pyramid_sizes = []

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
            [pyramid_sizes.append((int(H*s),int(W*s),int(C/s))) for s in scales]
            rescales = [
                scales[int(i / self.input_layers_num * 4)]
                for i in range(self.input_layers_num)
            ]

        print(pyramid_sizes)

        self.neck = Feature2Pyramid(
            embed_dim=self.in_channels,
            rescales=rescales,
        )

        self.align_corners = False

        self.in_channels = self.encoder.embed_dim
        self.embed_dims = self.encoder.embed_dim
        self.channels = channels
        self.num_classes = num_classes

        self.embed_dims = int(self.embed_dims / scales[-1])

        self.muster = MusterHead(
            embed_dims=self.embed_dims,
            channels=self.channels,
            patch_size=self.patch_size,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            depths=self.depths,
            num_heads=self.num_heads,
            strides=self.strides,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            act_cfg=self.act_cfg,
            norm_cfg=self.norm_cfg,
            init_cfg=None,
            pyramid_sizes=pyramid_sizes
        )

        self.muster_output_channels = int(self.in_channels / rescales[-3])

        self.reshape = ConvModule(
            in_channels=self.muster_output_channels,
            out_channels=self.channels,
            kernel_size=1,
            act_cfg=act_cfg
        )
        
        if self.interp_method == 'PixelShuffle':
            r = self.encoder.input_size // H
            self.conv_seg = nn.Conv2d(self.channels, self.num_classes*r*r, kernel_size=1)
            self.pixel_shuffle = PixelShuffle(r)
        elif self.interp_method == 'interpolate':
            self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        
        # If the encoder handles multi_temporal we feed it with the input
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
        
        if self.pyramid_strategy == 'head':
            feat = [feat[-1] for _ in range(len(self.pool_scales))]
            feat = self.neck(feat)
        elif self.pyramid_strategy == 'intermediate':
            feat = self.neck(feat)
        
        # head_input = []
        for i in range(len(feat)):
            f = feat[i]
            # print(f.shape)
            # f = rearrange(f,'B C H W -> B H W C')
            # head_input.append(f)

        # feat = head_input
        feat = self.muster(feat)

        feat = self.dropout(feat)
        feat = self.reshape(feat)
        output = self.conv_seg(feat)

        # fixed bug just for optical single modality
        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        # interpolate to the target spatial dims
        if self.interp_method == 'interpolate':
            output = F.interpolate(output, size=output_shape, mode="bilinear")
        elif self.interp_method == 'PixelShuffle':
            output = self.pixel_shuffle(output)

        return output

class WindowMSA(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        window_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop_rate=0.,
        proj_drop_rate=0.,
        init_cfg=None
    ):
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate

        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        super().__init__()

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)
        self.skip_qkv = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)
    def init_weights(self):
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    def forward(self, x, skip_x, mask=None):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C)
            skip_x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        assert x.shape == skip_x.shape, 'x.shape != skip_x.shape in WindowMSA'
        qkv = self.qkv(x).reshape(B, N, 2, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        skip_qkv = self.skip_qkv(skip_x).reshape(B, N, 1, self.num_heads,
                                            C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = skip_qkv[0], qkv[0], qkv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)
    
class ShiftWindowMSA(nn.Module):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 drop_prob=0,
                 dropout_layer = dict(type='DropPath',drop_prob=0.),
                 init_cfg=None):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = DropPath(drop_prob=drop_prob)


    def forward(self, query, skip_query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        
        # print(query.shape)
        # print(skip_query.shape)
        # print(hw_shape)
        assert L == H * W, 'input feature has wrong size'
        assert query.shape == skip_query.shape, f'skip query should has the same shape with query, found query shape {query.shape} and skip query shape {skip_query.shape}'
        query = query.view(B, H, W, C)
        skip_query = skip_query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        skip_query = F.pad(skip_query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
            shifted_skip_query = torch.roll(
                skip_query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            shifted_skip_query = skip_query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        skip_query_windows = self.window_partition(shifted_skip_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size ** 2, C)
        skip_query_windows = skip_query_windows.view(-1, self.window_size ** 2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, skip_query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(nn.Module):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):

        super(SwinBlock, self).__init__()

        self.skip_norm = build_norm_layer(norm_cfg, embed_dims)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

        self.norm3 = build_norm_layer(norm_cfg, embed_dims)

    def forward(self, x, skip_x, hw_shape):

        def _inner_forward(x, skip_x):
            identity = x
            # print(x.shape)
            skip_x = self.skip_norm(skip_x)
            x = self.norm1(x)
            x = self.attn(x, skip_x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            x = self.norm3(x)
            return x

        # if self.with_cp and x.requires_grad:
        #     x = cp.checkpoint(_inner_forward, x)
        # else:
        #     x = _inner_forward(x, skip_x)
        x = _inner_forward(x,skip_x)

        return x


class SwinBlockSequence(nn.Module):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
        is_upsample (bool): Whether to apply Fuse&Upsample block.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 is_upsample=False,
                 upsample_size=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(SwinBlockSequence,self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                init_cfg=None)
            self.blocks.append(block)

        self.is_upsample = is_upsample
        self.upsample_size = upsample_size
        self.conv = ConvModule(
            in_channels=embed_dims * 2,
            out_channels=embed_dims * 2,
            kernel_size=1,
            stride=1,
            norm_cfg=dict(type='SyncBN'),
            act_cfg=act_cfg)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x, skip_x, hw_shape):
        for block in self.blocks:
            x = block(x, skip_x, hw_shape)

        if self.is_upsample:
            x = torch.cat([x, skip_x], dim=2)
            up_hw_shape = [self.upsample_size[0],self.upsample_size[1]]
            B, HW, C = x.shape
            x = x.view(B, hw_shape[0], hw_shape[1], C)
            x = x.permute(0, 3, 1, 2)
            x = self.conv(x)
            x = self.ps(x)
            
            # print(f'Upsample Size: {self.upsample_size}')
            #correction for odd HW product
            if (self.upsample_size[0] % 2) != 0:
                x = F.interpolate(x, (self.upsample_size[0],self.upsample_size[1]),mode='bilinear')
            
            x = x.permute(0, 2, 3, 1).view(B, up_hw_shape[0] * up_hw_shape[1], C // 4)

            
            return x
        else:
            x = torch.cat([x, skip_x], dim=2)
            B, HW, C = x.shape
            x = x.view(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2)
            x = self.conv(x)
            x = x.permute(0, 2, 3, 1).view(B, hw_shape[0] * hw_shape[1], C)
            return x
        
class MusterHead(nn.Module):

    def __init__(self,
                 embed_dims=768,
                 channels = 128,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 pyramid_sizes=None, 
                 **kwargs):
        super().__init__()

        num_layers = len(depths)

        assert strides[3] == patch_size, 'Use non-overlapping patch embed.'

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = nn.ModuleList()
        in_channels = embed_dims
        self.channels = channels
        self.pyramid_sizes = pyramid_sizes

        for i in range(num_layers):
            if i < num_layers - 1:
                is_upsample = True
                upsample_size = self.pyramid_sizes[3-(i+1)]
            else:
                is_upsample = False
                upsample_size = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_channels),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                is_upsample=is_upsample,
                upsample_size=upsample_size,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                init_cfg=None)
            self.stages.append(stage)
            if is_upsample:
                in_channels = in_channels // 2

        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        self.mlp_ratio = mlp_ratio

        self.bottleneck_channels = in_channels*2

        self.ffns = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            mlp = FFN(
                embed_dims=in_channels,
                feedforward_channels=int(self.mlp_ratio * in_channels),
                num_fcs=2,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                add_identity=True,
                init_cfg=None)
            self.ffns.append(mlp)
            in_channels *= 2
        self.outffn = FFN(
            embed_dims=self.bottleneck_channels,
            feedforward_channels=int(self.mlp_ratio * self.channels),
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        # inputs = self._transform_inputs(inputs)

        inputs = list(inputs)
        # print(inputs[3].shape)
        B, C, H, W = inputs[3].shape
        
        shapes = [input.shape for input in inputs]
        
        hw_shape = (H, W)
        index = 0
        for ffn in self.ffns:
            ind = inputs[index]
            ind = ind.permute(0, 2, 3, 1)
            B, H, W, C = ind.shape
            hw_shape = (H, W)
            ind = ind.view(B, H * W, C)
            ind = ffn(ind)
            inputs[index] = ind
            index += 1

        x = inputs[3]
        for i, stage in enumerate(self.stages):
            C //= 2
            x = stage(x, inputs[3 - i], hw_shape)
            
            if i < len(self.stages) - 1:
                B, C, H, W = shapes[3-(i+1)]
                hw_shape = (H,W)

        out = x
        out = self.outffn(out)
        out = out.view(B, hw_shape[0], hw_shape[1], C * 4).permute(0, 3, 1, 2)
        # out = self.cls_seg(out)

        return out

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
    