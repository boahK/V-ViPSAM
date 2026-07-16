import torch, torch.nn as nn
from typing import Type
from einops import rearrange

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# VGCA
class CrossAttentionFusion(nn.Module):
    def __init__(self, c=256, heads=8, mlp_dim=1024):
        super().__init__()
        self.ct_ln = nn.LayerNorm(c)
        self.mri_ln = nn.LayerNorm(c)

        self.kv_pool = nn.AdaptiveMaxPool2d(16)
        self.kv_proj = nn.Conv2d(c, c, kernel_size=1, stride=1, bias=False)
        # self.kv_proj = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.kv_proj_norm = LayerNorm2d(c)

        self.mha = nn.MultiheadAttention(c, heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.1))
        
        self.mlp = MLPBlock(c, mlp_dim, act=nn.GELU)
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        
        self.residual_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, ct, mri):
        B, C, H, W = ct.shape
        q = ct.flatten(2).transpose(1, 2)

        mri_kv = self.kv_pool(mri)
        mri_kv = self.kv_proj_norm(self.kv_proj(mri_kv))
        k = mri_kv.flatten(2).transpose(1, 2)
        if not getattr(self, "_dbg_shapes_once", False):
            n_q = q.shape[1]
            n_kv = k.shape[1]
            _, _, h2, w2 = mri_kv.shape
            print(f"[CrossAttentionFusion] q_tokens={n_q} (H,W)=({H},{W}), "
                f"kv_tokens={n_kv} (H2,W2)=({h2},{w2})")
            self._dbg_shapes_once = True
        v = mri_kv.flatten(2).transpose(1, 2)
        
        q = self.ct_ln(q)
        k = self.mri_ln(k)
        v = self.mri_ln(v)
        
        attn_out, _ = self.mha(q, k, v)
        fused = self.norm1(q + self.gate * attn_out)
        
        mlp_out = self.mlp(fused)
        fused = self.norm2(fused + mlp_out)
        
        ct_flat = ct.flatten(2).transpose(1, 2)
        w = torch.sigmoid(self.residual_weight)
        fused = w * ct_flat + (1 - w) * fused
        
        return fused.transpose(1, 2).reshape(B, C, H, W)
