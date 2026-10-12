"""Conditional JiT model for 2D NCCT-to-CTA translation."""

import torch
import torch.nn as nn
from model_jit import (
    BottleneckPatchEmbed, TimestepEmbedder, 
    JiTBlock, FinalLayer, VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed
)


class JiT_CT2D(nn.Module):
    """Transformer that generates CTA slices from NCCT conditions."""
    def __init__(
        self,
        input_size=512,
        patch_size=16,
        in_channels=1,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        condition_channels=1,
        condition_fusion='concat',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.condition_channels = condition_channels
        self.condition_fusion = condition_fusion
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, 
            bottleneck_dim, hidden_size, bias=True
        )
        
        if condition_fusion == 'concat':
            self.condition_embedder = BottleneckPatchEmbed(
                input_size, patch_size, condition_channels,
                bottleneck_dim, hidden_size, bias=True
            )
            self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)
        elif condition_fusion == 'add':
            self.condition_embedder = BottleneckPatchEmbed(
                input_size, patch_size, condition_channels,
                bottleneck_dim, hidden_size, bias=True
            )
            self.fusion_scale = nn.Parameter(torch.ones(1))
        elif condition_fusion == 'cross_attn':
            self.condition_embedder = BottleneckPatchEmbed(
                input_size, patch_size, condition_channels,
                bottleneck_dim, hidden_size, bias=True
            )
        else:
            raise ValueError(f"Unknown condition_fusion: {condition_fusion}")
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), 
            requires_grad=False
        )
        
        grid_size = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            grid_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        grid_size = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            grid_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        for embedder in [self.x_embedder, self.condition_embedder]:
            w1 = embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(embedder.proj2.bias, 0)
        
        if self.condition_fusion == 'concat':
            nn.init.xavier_uniform_(self.fusion_layer.weight)
            nn.init.constant_(self.fusion_layer.bias, 0)
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def unpatchify(self, x, p):
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x, t, x_condition):
        t_emb = self.t_embedder(t)
        x_emb = self.x_embedder(x)
        x_emb += self.pos_embed
        
        cond_emb = self.condition_embedder(x_condition)
        cond_emb += self.pos_embed
        
        if self.condition_fusion == 'concat':
            x = torch.cat([x_emb, cond_emb], dim=-1)
            x = self.fusion_layer(x)
        elif self.condition_fusion == 'add':
            x = x_emb + self.fusion_scale * cond_emb
        elif self.condition_fusion == 'cross_attn':
            x = x_emb + 0.5 * cond_emb
        
        for block in self.blocks:
            x = block(x, t_emb, self.feat_rope)
        
        x = self.final_layer(x, t_emb)
        output = self.unpatchify(x, self.patch_size)
        return output


def JiT_CT2D_B_16(**kwargs):
    return JiT_CT2D(
        depth=12, hidden_size=768, num_heads=12,
        bottleneck_dim=128, patch_size=16, **kwargs
    )

def JiT_CT2D_B_32(**kwargs):
    return JiT_CT2D(
        depth=12, hidden_size=768, num_heads=12,
        bottleneck_dim=128, patch_size=32, **kwargs
    )

def JiT_CT2D_L_16(**kwargs):
    return JiT_CT2D(
        depth=24, hidden_size=1024, num_heads=16,
        bottleneck_dim=128, patch_size=16, **kwargs
    )


JiT_CT2D_models = {
    'JiT-CT2D-B/16': JiT_CT2D_B_16,
    'JiT-CT2D-B/32': JiT_CT2D_B_32,
    'JiT-CT2D-L/16': JiT_CT2D_L_16,
}
