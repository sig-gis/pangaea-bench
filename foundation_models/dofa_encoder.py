from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from timm.models.vision_transformer import Block

from .pos_embed import get_1d_sincos_pos_embed_from_grid_torch
from utils.registry import ENCODER_REGISTRY
from .base import Base_Encoder 


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, #enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is
        # too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)


    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[-1]
        )  # Using the last output to generate bias
        return weights, bias


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)
       

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class Dynamic_MLP_OFA(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()

        self.wv_planes = wv_planes
        self.inter_dim = inter_dim
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim      
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """
        initialize the base weights and dynamic mlp weights
        """
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs):
        inplanes = wvs.size(0)

        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000).float()
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  

        dynamic_weight = weight.view(
            self.embed_dim, inplanes, self.kernel_size, self.kernel_size
        ) 
        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


@ENCODER_REGISTRY.register()
class DOFA_Encoder(Base_Encoder):
    def __init__(self, 
                 cfg, 
                 img_size=224, 
                 patch_size=16, 
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16, 
                 wv_planes=128, 
                 # global_pool=True,
                 return_all_tokens = True,
                 mlp_ratio=4., 
                 use_norm=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.model_name = 'DOFA'
        self.output_layers = cfg['output_layers']
        self.img_size = img_size
        self.wv_planes = wv_planes
        # self.global_pool = global_pool
        self.return_all_tokens = return_all_tokens
        self.embed_dim = embed_dim 
        self.patch_size = patch_size
        self.use_norm = use_norm
        self.input_bands = cfg.get("input_bands")
        self.wv_list=[cfg['wave_list'][m][bi] for m, b in self.input_bands.items() for bi in b ]

        # if self.global_pool:
        #     norm_layer = norm_layer
        #     embed_dim = embed_dim
        #     self.fc_norm = norm_layer(embed_dim)
        # else:
        #     self.norm = norm_layer(embed_dim)
        self.norm = norm_layer([embed_dim, (img_size // patch_size) ,(img_size // patch_size)])

        
        self.patch_embed = Dynamic_MLP_OFA(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        

    def forward(self, image):
        # embed patches
        x = image['optical']
        wavelist = torch.tensor(self.wv_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                out = x[:, 1:].permute(0, 2, 1).view(x.shape[0], -1, self.img_size // self.patch_size, self.img_size // self.patch_size).contiguous()
                if self.use_norm:
                    out = self.norm(out)
                output.append(out)

        return output
