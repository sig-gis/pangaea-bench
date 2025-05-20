import torch
from torch import nn
import math

from pangaea.encoders.pos_embed import get_2d_sincos_pos_embed_with_resolution
from pangaea.encoders.anysat_utils.transformers import BlockTransformer, trunc_normal_


class TransformerMulti(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        input_res={},
        modalities={},
        scales={},
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.input_res = input_res
        datasets = list(scales.keys())
        self.predictor_pos_embed = {}
        for dataset in datasets:
            for scale in scales[dataset]:
                self.predictor_pos_embed['_'.join([dataset, str(scale)])] = get_2d_sincos_pos_embed_with_resolution(
                                                                                embed_dim,
                                                                                scale,
                                                                                input_res,
                                                                                cls_token=True,
                                                                                modalities=modalities[dataset]
                                                                            )
        # --
        self.predictor_blocks = nn.ModuleList([
            BlockTransformer(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=None, n_modalities=1,
                drop=0., attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, modality, dataset, scale, keep_subpatch=False):
        # -- concat class token to x
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        # -- add positional embedding to x tokens
        x += self.predictor_pos_embed['_'.join([dataset, str(scale)])][modality].to(x.device)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        if keep_subpatch:
            return x[:, 0], x[:, 1:]
        
        return x[:, 0]
    
    def forward_release(self, x, modality, scale, keep_subpatch=False):
        B, N, C = x.shape
        # -- concat class token to x
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        # -- add positional embedding to x tokens
        x += get_2d_sincos_pos_embed_with_resolution(C,
                                                        scale,
                                                        self.input_res,
                                                        cls_token=True,
                                                        modalities=[modality]
                                                    )[modality].to(x.device)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        if keep_subpatch:
            return x[:, 0], x[:, 1:]
        
        return x[:, 0]

