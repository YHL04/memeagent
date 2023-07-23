

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class GTrXL(nn.Module):

    def __init__(self, C, H, W,
                 nlayers, dim, nhead, p,
                 device="cuda"):
        super(GTrXL, self).__init__()
        self.nlayers = nlayers
        self.dim = dim
        self.device = device

        self.proj = PatchProjection(
            C=C,
            H=H,
            W=W,
            dim=dim,
            patch_size=20,
        )

        self.layers = nn.ModuleList([
            AttentionLayer(
                dim=dim,
                nhead=nhead,
                p=p
            )
            for _ in range(nlayers)])

    def forward(self, x):
        # project patches into embeddings
        x = self.proj(x)

        # pass through gated layers
        for layer in self.layers:
            x = layer(x)

        # get embedding from CLS token
        x = x[:, 0, :]

        return x


class PatchProjection(nn.Module):

    def __init__(self, C, H, W, dim, patch_size):
        super(PatchProjection, self).__init__()

        self.C = C
        self.H = H
        self.W = W
        self.dim = dim
        self.patch_size = patch_size

        self.nb_patch = (self.H // self.patch_size) * (self.W // self.patch_size)

        self.patch_emb = nn.Conv2d(
            in_channels=self.c,
            out_channels=self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.nb_patch+1, self.dim)
        )
        self.cls = nn.Parameter(
            torch.randn(1, 1, self.dim)
        )

    def forward(self, x):
        # (B, C, H, W)
        x = self.patch_emb(x)
        # (B, D, H/P, W/P)
        x = torch.flatten(x, 2, 3).transpose(1, 2)
        # (B, N * P, D)
        x = torch.cat([self.cls.repeat(x.size(0), 1, 1), x], dim=1)
        # (B, N*P + 1, D)

        return x + self.pos_emb


class Attention(nn.Module):

    def __init__(self, dim, nhead):
        super(Attention, self).__init__()
        self.nhead = nhead

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.concat = nn.Linear(dim, dim, bias=False)

    def forward(self, q, kv, mask=None):
        q, k, v = self.q(q), *self.kv(kv).chunk(2, dim=-1)
        q = rearrange('b l (h d) -> b h l d', q, h=self.nhead)
        k, v = k.unsqueeze(1), v.unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = self.concat(out)
        out = rearrange('b h l d -> b l (h d)', out)

        return out


class FeedForward(nn.Module):
    """
    A simple feed forward network to be used in transformer layers.

    Architecture:
        Sequential(
            LayerNorm(dim)
            Linear(dim, inner_dim)
            GELU()
            Linear(inner_dim, dim)
        )

    Args:
        dim (int): The dimension of the input and output
        hidden_dim (int): The dimension of the hidden layer
    """

    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)


class GRUGate(nn.Module):
    """
    GRU Gating for Gated Transformer-XL (GTrXL)

    See Stabilizing Transformer for Reinforcement Learning:
    https://arxiv.org/pdf/1910.06764v1.pdf
    """

    def __init__(self, dim, bg=0.1):
        super(GRUGate, self).__init__()
        self.Wr = nn.Linear(dim, dim)
        self.Ur = nn.Linear(dim, dim)
        self.Wz = nn.Linear(dim, dim)
        self.Uz = nn.Linear(dim, dim)
        self.Wg = nn.Linear(dim, dim)
        self.Ug = nn.Linear(dim, dim)
        self.bg = bg

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)

        return g


class AttentionLayer(nn.Module):

    def __init__(self, dim, nhead, p):
        self.attention = Attention(dim=dim, nhead=nhead)
        self.norm1 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn = FeedForward(dim=dim, hidden_dim=4 * dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, x, mask):
        _x = x
        x = self.norm1(x)
        x = self.attention(q=x, kv=x, mask=mask)

        x = self.gate1(_x, x)
        x = self.dropout1(x)

        _x = x
        x = self.norm2(x)
        x = self.ffn(x)

        x = self.gate2(_x, x)
        x = self.dropout2(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Compute sinusoid encoding from original transformer paper.

    Args:
        dim (int): dimension of model
        seqlen (int): max length of transformer
    """
    def __init__(self, dim, seqlen, device):
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(seqlen, dim, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, seqlen, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, dim, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / dim)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        """Obtain positional encoding according to input size"""
        batch_size, seq_len, d_model = x.size()
        return self.encoding[:seq_len, :]
