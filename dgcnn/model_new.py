import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Canonicalizer:
    """
    Contract: All canonicalizers must return (canonical_pc, perm, R), where `perm` is
    strictly the argsort index array used to form `canonical_pc` via `torch.gather`.
    """

    @staticmethod
    def center_and_normalize(pc, eps=1e-8):
        pc = pc - pc.mean(dim=1, keepdim=True)
        max_norm = torch.max(torch.linalg.norm(pc, dim=2), dim=1, keepdim=True)[0]
        pc = pc / torch.clamp(max_norm, min=eps).unsqueeze(-1)
        return pc

    @staticmethod
    def _enforce_so3(R):
        det = torch.linalg.det(R)
        flip = (det < 0).to(R.dtype).view(-1, 1, 1)
        Ffix = torch.eye(3, device=R.device, dtype=R.dtype).view(1, 3, 3).repeat(R.shape[0], 1, 1)
        Ffix[:, 2, 2] = 1.0 - 2.0 * flip.squeeze(-1).squeeze(-1)
        return torch.bmm(R, Ffix)

    @staticmethod
    def _fix_eig_signs(vecs):
        B, N, K = vecs.shape
        max_idx = torch.argmax(vecs.abs(), dim=1, keepdim=True)
        max_vals = torch.gather(vecs, 1, max_idx)
        signs = torch.sign(max_vals)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return vecs * signs

    @staticmethod
    def get_fiedler_permutation(pc, sigma_kernel=3.0, epsilon=1e-8):
        B, N, D = pc.shape
        device = pc.device
        dist_sq = torch.cdist(pc, pc, p=2).pow(2)
        W = torch.exp(-dist_sq / (sigma_kernel ** 2))
        W.diagonal(dim1=-2, dim2=-1).fill_(0)
        D_vec = W.sum(dim=2) + epsilon
        D_inv_sqrt = torch.rsqrt(D_vec)
        W_norm = W * D_inv_sqrt.unsqueeze(1) * D_inv_sqrt.unsqueeze(2)
        I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        L_sym = I - W_norm

        vals, vecs = torch.linalg.eigh(L_sym)
        fiedler = vecs[:, :, 1]

        centered_fied = fiedler - torch.mean(fiedler, dim=1, keepdim=True)
        skew = (centered_fied ** 3).mean(dim=1, keepdim=True)
        signs = torch.sign(skew)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        fiedler = fiedler * signs

        sorted_fiedler, perm = torch.sort(fiedler, dim=1)
        return perm

    @staticmethod
    def spectral_fiedler(pc, sigma_kernel=1.0, epsilon=1e-8):
        B, N, D = pc.shape
        device = pc.device

        perm = Canonicalizer.get_fiedler_permutation(pc, sigma_kernel=sigma_kernel, epsilon=epsilon)
        pc_ord = torch.gather(pc, 1, perm.unsqueeze(-1).expand(-1, -1, D))

        weights = torch.linspace(-1, 1, N, device=device).view(1, N, 1)
        moment_1 = (pc_ord * weights).sum(dim=1, keepdim=True)
        moment_2 = (pc_ord * (weights ** 2)).sum(dim=1, keepdim=True)
        moment_3 = torch.cross(moment_1, moment_2, dim=2)

        u1 = F.normalize(moment_1, dim=2, eps=epsilon)
        u2_proj = moment_2 - (moment_2 * u1).sum(dim=2, keepdim=True) * u1
        u2 = F.normalize(u2_proj, dim=2, eps=epsilon)
        u3 = F.normalize(moment_3, dim=2, eps=epsilon)

        R = torch.stack([u1.squeeze(1), u2.squeeze(1), u3.squeeze(1)], dim=-1)
        R = Canonicalizer._enforce_so3(R)
        canonical_pc = torch.bmm(pc_ord, R)

        return canonical_pc, perm, R

    @staticmethod
    def _apply_data_signs(canonical_pc, R, s):
        s = torch.where(s == 0, torch.ones_like(s), s)
        flips = (s < 0).sum(dim=-1)
        odd = (flips % 2 == 1).view(-1)
        s[odd, 2] *= -1
        canonical_pc2 = canonical_pc * s.unsqueeze(1)
        R2 = Canonicalizer._enforce_so3(R * s.unsqueeze(1))
        return canonical_pc2, R2

    @staticmethod
    def _order(canonical_pc):
        B, N, D = canonical_pc.shape
        device = canonical_pc.device

        perm = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        for dim_idx in (2, 1, 0):
            vals = torch.gather(canonical_pc[..., dim_idx], 1, perm)
            sort_idx = torch.argsort(vals, dim=1, stable=True)
            perm = torch.gather(perm, 1, sort_idx)

        ordered = torch.gather(canonical_pc, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        return ordered, perm

    @staticmethod
    def pca_skew(pc, epsilon=1e-8):
        B, N, D = pc.shape

        cov = torch.bmm(pc.transpose(1, 2), pc) / (N - 1)
        cov += torch.eye(D, device=pc.device).unsqueeze(0) * epsilon

        _, eigenvectors = torch.linalg.eigh(cov)
        eigenvectors = eigenvectors.flip(dims=[2])
        eigenvectors = Canonicalizer._fix_eig_signs(eigenvectors)
        eigenvectors = Canonicalizer._enforce_so3(eigenvectors)

        canonical_pc = torch.bmm(pc, eigenvectors)
        skew = (canonical_pc ** 3).mean(dim=1)
        s = torch.sign(skew)

        canonical_pc2, R_final = Canonicalizer._apply_data_signs(canonical_pc, eigenvectors, s)
        ordered, perm = Canonicalizer._order(canonical_pc2)

        return ordered, perm, R_final


class CanonicalizationWrapper(nn.Module):
    def __init__(self, method='pca'):
        super().__init__()
        self.method = method.lower()

    @torch.no_grad()
    def forward(self, x):
        # FIX: Correctly detect if channels are in the middle (B, 3, N)
        if x.shape[1] == 3 and x.shape[2] > 3:
            x = x.transpose(1, 2).contiguous()

        xyz = x[..., :3]
        features = x[..., 3:] if x.shape[-1] > 3 else None

        xyz = Canonicalizer.center_and_normalize(xyz)

        if self.method == 'pca':
            xyz_canon, perm, R_final = Canonicalizer.pca_skew(xyz)
        elif self.method == 'spectral':
            xyz_canon, perm, R_final = Canonicalizer.spectral_fiedler(xyz)
        else:
            raise ValueError(f"Unknown canonicalization method: {self.method}")

        if features is not None:
            features_ord = torch.gather(features, 1, perm.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
            out = torch.cat([xyz_canon, features_ord], dim=-1)
        else:
            out = xyz_canon

        return out


# ===========================================================================
#  Model Baseline: Transformer with Shared Relative Position Encoding
# ===========================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class RelativePositionEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, num_heads=6):
        super().__init__()
        # in_dim is 4 because we input (diff_x, diff_y, diff_z, distance)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads)
        )

        # Calibrate the last layer to ensure positional bias doesn't dominate early training
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        if self.mlp[-1].bias is not None:
            nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, rpe_features):
        pos_enc = self.mlp(rpe_features)
        # Explicit contiguous layout prevents implicit copying overhead
        return pos_enc.permute(0, 3, 1, 2).contiguous()


class GeometricAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"Attention dimension ({dim}) must be divisible by num_heads ({num_heads})."

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rpe_bias):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + rpe_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Transpose -> contiguous -> view is explicitly safe and optimal
        x = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GeometricAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, rpe_bias):
        x = x + self.drop_path(self.attn(self.norm1(x), rpe_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PointTransformerClassifier(nn.Module):
    def __init__(self,
                 num_classes=40,
                 in_channels=3,
                 canon_method='pca',
                 dim=216,
                 depth=4,
                 heads=6,
                 mlp_ratio=2.0,
                 drop_rate=0.1,
                 drop_path_rate=0.1):
        super().__init__()
        self.canon = CanonicalizationWrapper(method=canon_method)

        # RPE now accepts 4 input features: (x, y, z, L2_distance)
        self.rpe = RelativePositionEncoder(in_dim=4, hidden_dim=64, num_heads=heads)

        self.embedding = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.GELU(),
            nn.Linear(64, dim)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=drop_rate,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.canon(x)

        xyz = x[..., :3]

        # Explicit Euclidean distance calculation
        diff = xyz.unsqueeze(2) - xyz.unsqueeze(1)

        # We must add an epsilon to avoid NaNs during backprop when diff is 0 (the diagonal)
        dist = torch.sqrt((diff ** 2).sum(dim=-1, keepdim=True) + 1e-8)

        rpe_features = torch.cat([diff, dist], dim=-1)
        rpe_bias = self.rpe(rpe_features)

        x = self.embedding(x)

        for block in self.blocks:
            x = block(x, rpe_bias)

        x = self.norm(x)
        x = x.max(dim=1)[0]
        return self.head(x)
# ===========================================================================
#  Model 1: The Global MLP Baseline (Modularized)
# ===========================================================================

class FourierFeatureMap(nn.Module):
    def __init__(self, in_features=3, num_bands=4, scale=10.0):
        super().__init__()
        self.register_buffer('B_matrix', torch.randn(in_features, num_bands * in_features) * scale)

    def forward(self, x):
        x_proj = (2. * math.pi * x) @ self.B_matrix
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLPResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.drop(self.act(self.linear1(self.norm(x))))


class GlobalMLPClassifier(nn.Module):
    def __init__(self,
                 num_classes=40,
                 num_points=1024,
                 in_channels=3,
                 canon_method='pca',
                 num_bands=4,
                 mlp_dims=[512, 256, 128],
                 dropout_rates=[0.5, 0.5, 0.3]):
        super().__init__()
        assert len(mlp_dims) == len(dropout_rates), "mlp_dims and dropout_rates must match in length."

        self.num_points = num_points
        self.canon = CanonicalizationWrapper(method=canon_method)
        self.fourier_map = FourierFeatureMap(in_features=in_channels, num_bands=num_bands, scale=10.0)

        self.input_dim = num_points * (in_channels * num_bands * 2)

        blocks = []
        current_dim = self.input_dim
        for dim, drop in zip(mlp_dims, dropout_rates):
            blocks.append(MLPResidualBlock(current_dim, dim, drop=drop))
            current_dim = dim

        self.blocks = nn.Sequential(*blocks)
        self.final_norm = nn.LayerNorm(current_dim)
        self.head = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        # Move canonicalization to the top to standardize shape to (B, N, 3)
        x = self.canon(x)

        B, N, _ = x.shape
        assert N == self.num_points, f"GlobalMLP strictly requires N={self.num_points} points, got {N}."

        x = self.fourier_map(x)
        x = x.view(B, -1)

        x = self.blocks(x)
        x = self.final_norm(x)
        return self.head(x)


