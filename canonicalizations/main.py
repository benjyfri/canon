import torch
import torch.nn.functional as F
import math
import random
import numpy as np
import time
import h5py
import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
# ==========================================
# 1. Modular Data Provider
# ==========================================

import h5py
import glob
import os
import plotly.graph_objects as go
import torch
import numpy as np

def plot_alignment_comparison(pc1, pc2, title="Alignment Comparison"):
    if isinstance(pc1, torch.Tensor): pc1 = pc1.detach().cpu().numpy()
    if isinstance(pc2, torch.Tensor): pc2 = pc2.detach().cpu().numpy()
    if pc1.ndim == 3: pc1 = pc1[0]
    if pc2.ndim == 3: pc2 = pc2[0]

    fig = go.Figure()

    # Original Cloud (Blue)
    fig.add_trace(go.Scatter3d(
        x=pc1[:, 0], y=pc1[:, 1], z=pc1[:, 2],
        mode='markers', name='Canonical Original',
        marker=dict(size=3, color='blue', opacity=0.6)
    ))

    # Transformed/Noisy Cloud (Red)
    fig.add_trace(go.Scatter3d(
        x=pc2[:, 0], y=pc2[:, 1], z=pc2[:, 2],
        mode='markers', name='Canonical Noisy',
        marker=dict(size=3, color='red', opacity=0.6)
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()
def plot_interactive_point_cloud(pc, title="Point Cloud", marker_size=3, color='royalblue'):
    """
    Renders an interactive 3D plot of a point cloud.
    Accepts either a NumPy array or a PyTorch tensor of shape (N, 3) or (1, N, 3).
    """
    # 1. Handle PyTorch tensors safely
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()

    # 2. Handle accidental batch dimensions (B, N, 3) -> (N, 3)
    if pc.ndim == 3:
        if pc.shape[0] == 1:
            pc = pc[0]
        else:
            print(f"Warning: Received batch of {pc.shape[0]} point clouds. Plotting the first one.")
            pc = pc[0]

    if pc.shape[1] != 3:
        raise ValueError(f"Expected point cloud with 3 coordinates per point, got shape {pc.shape}")

    # 3. Extract coordinates
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

    # 4. Build the Plotly figure
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=color,
            opacity=0.8,
            line=dict(width=0)  # Removes borders for a cleaner look
        )
    )])

    # 5. Format the layout for geometry
    fig.update_layout(
        title=dict(text=title, x=0.5),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(title='X', showbackground=False),
            yaxis=dict(title='Y', showbackground=False),
            zaxis=dict(title='Z', showbackground=False),
            aspectmode='data'  # CRITICAL: preserves true geometric proportions
        )
    )

    fig.show()

class PointCloudProvider:
    """
    Abstracts data loading for Synthetic and ModelNet40 H5 files.
    """

    def __init__(self, dataset_type='synthetic', batch_size=64, num_points=100, device='cpu', data_dir=None):
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_points = num_points
        self.device = device
        self.data_dir = data_dir

        if self.dataset_type == 'modelnet40':
            self._load_modelnet40()

    def _load_modelnet40(self):
        if self.data_dir is None:
            raise ValueError("data_dir must be provided for ModelNet40.")

        h5_files = glob.glob(os.path.join(self.data_dir, 'ply_data_train*.h5'))
        if not h5_files:
            raise FileNotFoundError(f"No train .h5 files found in {self.data_dir}. Check the path.")

        print(f"Loading ModelNet40 train data from {len(h5_files)} files...")
        all_data = []
        for f in h5_files:
            with h5py.File(f, 'r') as h5_f:
                all_data.append(h5_f['data'][:])

        all_data = np.concatenate(all_data, axis=0)
        self.modelnet_data = torch.tensor(all_data, dtype=torch.float32)
        self.num_total_shapes = self.modelnet_data.shape[0]
        print(f"Successfully loaded {self.num_total_shapes} shapes.\n")

    @staticmethod
    def _farthest_point_sample(xyz, npoint):
        """
        Batched Farthest Point Sampling in pure PyTorch.
        xyz: (B, N, 3) tensor
        npoint: int, number of points to sample
        Returns: (B, npoint, 3) sampled tensor
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10

        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)

            mask = dist < distance
            distance[mask] = dist[mask]

            farthest = torch.max(distance, -1)[1]

        sampled_clouds = torch.gather(xyz, 1, centroids.unsqueeze(-1).expand(-1, -1, 3))
        return sampled_clouds

    def get_batch(self):
        if self.dataset_type == 'synthetic':
            return self._generate_synthetic()
        elif self.dataset_type == 'modelnet40':
            return self._get_modelnet40_batch()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _generate_synthetic(self):
        clouds = torch.randn(self.batch_size, self.num_points, 3, device=self.device)
        clouds = F.normalize(clouds, p=2, dim=2)
        radii = torch.rand(self.batch_size, self.num_points, 1, device=self.device) ** (1 / 3)
        clouds = clouds * radii
        scales = torch.empty(self.batch_size, 1, 3, device=self.device)
        scales[:, 0, 0] = torch.rand(self.batch_size, device=self.device) * 1.5 + 1.5
        scales[:, 0, 1] = torch.rand(self.batch_size, device=self.device) * 0.7 + 0.8
        scales[:, 0, 2] = torch.rand(self.batch_size, device=self.device) * 0.4 + 0.1
        clouds = clouds * scales
        centroid = clouds.mean(dim=1, keepdim=True)
        return clouds - centroid

    def _get_modelnet40_batch(self):
        idx = torch.randint(0, self.num_total_shapes, (self.batch_size,))
        clouds = self.modelnet_data[idx].to(self.device)

        sampled_clouds = self._farthest_point_sample(clouds, self.num_points)

        centroid = sampled_clouds.mean(dim=1, keepdim=True)
        return sampled_clouds - centroid

    def apply_transforms(self, clouds, noise_std=0.01):
        B, N, _ = clouds.shape
        R_batch = generate_random_rotations(B, self.device)
        translation = (torch.rand(B, 1, 3, device=self.device) - 0.5) * 10.0
        transformed = torch.bmm(clouds, R_batch.transpose(1, 2)) + translation
        noise = torch.randn_like(transformed) * noise_std
        return transformed + noise
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_random_rotations(B, device):
    A = torch.randn(B, 3, 3, device=device)
    Q, R_mat = torch.linalg.qr(A)
    # Fix: Safe diagonal sign extraction
    d = torch.sign(torch.diagonal(R_mat, dim1=-2, dim2=-1))
    d = torch.where(d == 0, torch.ones_like(d), d)
    Q = Q * d.unsqueeze(-2)
    det = torch.linalg.det(Q)
    flip_mask = (det < 0)
    Q[flip_mask, :, 0] *= -1
    return Q


class Canonicalizer:
    # --- Shared Helpers ---
    @staticmethod
    def _center(pc):
        return pc - pc.mean(dim=1, keepdim=True)

    @staticmethod
    def _safe_normalize(v, eps=1e-8):
        return v / (v.norm(dim=-1, keepdim=True) + eps)

    @staticmethod
    def _enforce_so3(R):
        det = torch.linalg.det(R)
        flip = (det < 0).to(R.dtype).view(-1, 1, 1)
        Ffix = torch.eye(3, device=R.device, dtype=R.dtype).view(1, 3, 3).repeat(R.shape[0], 1, 1)
        Ffix[:, 2, 2] = 1.0 - 2.0 * flip.squeeze(-1).squeeze(-1)
        return torch.bmm(R, Ffix)

    @staticmethod
    def _fix_eig_signs(vecs):
        """Fixes eigenvector signs deterministically using the max absolute component."""
        B, N, K = vecs.shape
        max_idx = torch.argmax(vecs.abs(), dim=1, keepdim=True)
        max_vals = torch.gather(vecs, 1, max_idx)
        signs = torch.sign(max_vals)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return vecs * signs

    @staticmethod
    def _frame_from_two(u1, u2, eps=1e-8):
        B = u1.shape[0]
        device, dtype = u1.device, u1.dtype
        e1 = Canonicalizer._safe_normalize(u1, eps)
        u2o = u2 - (u2 * e1).sum(dim=-1, keepdim=True) * e1
        n2 = u2o.norm(dim=-1, keepdim=True)

        # Fix: Safe boolean broadcasting for fallback vectors
        use_a = (e1[:, 0].abs() < 0.9).unsqueeze(-1)
        a = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 3).expand(B, 3)
        b = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3).expand(B, 3)
        aux = torch.where(use_a, a, b)

        u2_fallback = torch.cross(e1, aux, dim=-1)
        e2 = Canonicalizer._safe_normalize(torch.where(n2 > 1e-6, u2o, u2_fallback), eps)
        e3 = Canonicalizer._safe_normalize(torch.cross(e1, e2, dim=-1), eps)
        R = torch.stack([e1, e2, e3], dim=-1)  # Columns are axes
        return Canonicalizer._enforce_so3(R)

    @staticmethod
    def _skew_sign_fix(canonical_pc, R, eps=1e-8):
        skew = (canonical_pc ** 3).mean(dim=1, keepdim=True)
        s = torch.sign(skew)
        s = torch.where(s == 0, torch.ones_like(s), s)

        # Fix: Ensure 'odd' is strictly a 1D boolean mask of shape (B,)
        flips = (s < 0).sum(dim=-1)  # shape: (B, 1)
        odd = (flips % 2 == 1).view(-1)  # shape: (B,)
        s[odd, 0, 2] *= -1

        canonical_pc2 = canonical_pc * s
        R2 = R * s  # (B, 3, 3) * (B, 1, 3) broadcasts perfectly to flip columns
        R2 = Canonicalizer._enforce_so3(R2)
        return canonical_pc2, R2
    @staticmethod
    def _order(canonical_pc):
        keys = canonical_pc[:, :, 0] + 1e-3 * canonical_pc[:, :, 1] + 1e-6 * canonical_pc[:, :, 2]
        perm = torch.argsort(keys, dim=1)
        ordered = torch.gather(canonical_pc, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
        return ordered, perm

    @staticmethod
    def _gather_points(pc, idx_bn):
        return torch.gather(pc, 1, idx_bn.unsqueeze(-1).expand(-1, -1, 3))

    @staticmethod
    def get_fiedler_permutation(pc, sigma_kernel=3.0, epsilon=1e-8, return_vals=False):
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
        # Fix: Enforce stable eigenvector signs, drop D_inv_sqrt multiplication to preserve norm
        vecs = Canonicalizer._fix_eig_signs(vecs)
        fiedler = vecs[:, :, 1]

        sorted_fiedler, perm = torch.sort(fiedler, dim=1)
        if return_vals: return perm, sorted_fiedler
        return perm

    # ==========================
    # ORIGINAL METHODS
    # ==========================
    @staticmethod
    def old_method(pc, epsilon=1e-10):
        # Rewritten to enforce true SE(3) equivariance using intrinsic data
        B, N, D = pc.shape
        device = pc.device
        centered = Canonicalizer._center(pc)
        perm = Canonicalizer.get_fiedler_permutation(centered)
        data = torch.gather(centered, 1, perm.unsqueeze(-1).expand(-1, -1, 3))

        m_point = torch.mean(data, dim=1)
        u1 = F.normalize(m_point, dim=1, eps=epsilon)

        # Use furthest point structurally to derive u2 without global anchors
        d_mpoint = torch.sum((data - m_point.unsqueeze(1)) ** 2, dim=2)
        idx_far = torch.argmax(d_mpoint, dim=1)
        p_far = Canonicalizer._gather_points(data, idx_far.unsqueeze(1)).squeeze(1)

        v2 = p_far - (torch.sum(p_far * u1, dim=1, keepdim=True) * u1)
        u2 = F.normalize(v2, dim=1, eps=epsilon)
        u3 = torch.cross(u1, u2, dim=1)

        R = torch.stack([u1, u2, u3], dim=-1)  # Columns are axes
        R = Canonicalizer._enforce_so3(R)
        final_points = torch.bmm(data, R)

        # Preserve original zero-out masking behavior
        mask = torch.ones_like(final_points)
        mask[:, -1, 1] = 0.0
        return final_points * mask, perm

    @staticmethod
    def pca_skew(pc, epsilon=1e-8):
        B, N, D = pc.shape
        centered = Canonicalizer._center(pc)
        cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)
        _, eigenvectors = torch.linalg.eigh(cov)
        eigenvectors = eigenvectors.flip(dims=[2])

        # Fix: apply deterministic eigenvector sign constraints
        eigenvectors = Canonicalizer._fix_eig_signs(eigenvectors)
        eigenvectors = Canonicalizer._enforce_so3(eigenvectors)

        canonical_pc = torch.bmm(centered, eigenvectors)  # Columns are axes
        canonical_pc, _ = Canonicalizer._skew_sign_fix(canonical_pc, eigenvectors)
        return Canonicalizer._order(canonical_pc)

    @staticmethod
    def spectral_fiedler(pc, sigma_kernel=1.0, epsilon=1e-8):
        B, N, D = pc.shape
        device = pc.device
        centered = Canonicalizer._center(pc)
        perm = Canonicalizer.get_fiedler_permutation(centered)
        pc_ord = torch.gather(centered, 1, perm.unsqueeze(-1).expand(-1, -1, D))

        weights = torch.linspace(-1, 1, N, device=device).view(1, N, 1)
        moment_1 = (pc_ord * weights).sum(dim=1, keepdim=True)
        moment_2 = (pc_ord * (weights ** 2)).sum(dim=1, keepdim=True)
        moment_2 = moment_2 + torch.randn_like(moment_2) * epsilon
        moment_3 = torch.cross(moment_1, moment_2, dim=2)

        u1 = F.normalize(moment_1, dim=2, eps=epsilon)
        u2_proj = moment_2 - (moment_2 * u1).sum(dim=2, keepdim=True) * u1
        u2 = F.normalize(u2_proj, dim=2, eps=epsilon)
        u3 = F.normalize(moment_3, dim=2, eps=epsilon)

        # Fix: Stack along column dimension to create proper R matrix
        R = torch.stack([u1.squeeze(1), u2.squeeze(1), u3.squeeze(1)], dim=-1)
        R = Canonicalizer._enforce_so3(R)
        canonical_pc = torch.bmm(pc_ord, R)
        return canonical_pc, perm

    # ==========================
    # SET A: GEOMETRIC & SPECTRAL
    # ==========================
    @staticmethod
    def ica_kurtosis(pc, epsilon=1e-8):
        B, N, D = pc.shape
        centered = Canonicalizer._center(pc)
        cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        W = eigvecs @ torch.diag_embed(torch.rsqrt(eigvals + epsilon))
        whitened = torch.bmm(centered, W)
        x2 = whitened ** 2

        C = torch.zeros((B, D, D), device=pc.device)
        for i in range(D):
            for j in range(D):
                C[:, i, j] = (x2[:, :, i] * x2[:, :, j]).mean(dim=1)
        C = C - torch.eye(D, device=pc.device).unsqueeze(0)
        _, k_vecs = torch.linalg.eigh(C)
        transform = W @ k_vecs

        # Fix: Project arbitrary transform back onto SO(3) via SVD to prevent geometric distortion
        U_svd, _, Vh_svd = torch.linalg.svd(transform)
        R = torch.bmm(U_svd, Vh_svd)
        R = Canonicalizer._enforce_so3(R)

        canonical_pc = torch.bmm(centered, R)
        return Canonicalizer._order(canonical_pc)

    @staticmethod
    def farthest_pair(pc, epsilon=1e-8):
        B, N, D = pc.shape
        device = pc.device
        centered = Canonicalizer._center(pc)

        # Fix: Permutation invariant initialization
        centroid_dist = torch.sum(centered ** 2, dim=2)
        idx_first = torch.argmax(centroid_dist, dim=1)
        first = Canonicalizer._gather_points(centered, idx_first.unsqueeze(1)).squeeze(1)

        d0 = torch.sum((centered - first.unsqueeze(1)) ** 2, dim=2)
        idx_i = torch.argmax(d0, dim=1)
        pi = Canonicalizer._gather_points(centered, idx_i.unsqueeze(1)).squeeze(1)

        d1 = torch.sum((centered - pi.unsqueeze(1)) ** 2, dim=2)
        idx_j = torch.argmax(d1, dim=1)
        pj = Canonicalizer._gather_points(centered, idx_j.unsqueeze(1)).squeeze(1)

        u1 = F.normalize(pj - pi, dim=1, eps=epsilon)
        proj = (centered @ u1.unsqueeze(-1)).squeeze(-1)
        perp = centered - proj.unsqueeze(-1) * u1.unsqueeze(1)
        idx_k = torch.argmax(torch.sum(perp ** 2, dim=2), dim=1)
        pk = Canonicalizer._gather_points(centered, idx_k.unsqueeze(1)).squeeze(1)

        v2 = pk - pi - (torch.sum((pk - pi) * u1, dim=1, keepdim=True) * u1)
        n2 = torch.norm(v2, dim=1, keepdim=True)

        # Fix: Stable collinear fallback
        a = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=pc.dtype).expand(B, 3)
        b = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=pc.dtype).expand(B, 3)
        use_a = (u1[:, 0].abs() < 0.9).unsqueeze(-1)
        aux = torch.where(use_a, a, b)
        u2_fallback = torch.cross(u1, aux, dim=1)

        v2 = torch.where(n2 > 1e-6, v2, u2_fallback)
        u2 = F.normalize(v2, dim=1, eps=epsilon)
        u3 = torch.cross(u1, u2, dim=1)

        R = torch.stack([u1, u2, u3], dim=-1)
        R = Canonicalizer._enforce_so3(R)
        canonical_pc = torch.bmm(centered, R)  # Standard column projection
        return Canonicalizer._order(canonical_pc)

    @staticmethod
    def max_norm(pc, epsilon=1e-8):
        B, N, D = pc.shape
        device = pc.device
        centered = Canonicalizer._center(pc)
        norms = torch.sum(centered ** 2, dim=2)
        idx = torch.argmax(norms, dim=1)
        p_max = Canonicalizer._gather_points(centered, idx.unsqueeze(1)).squeeze(1)
        u1 = F.normalize(p_max, dim=1, eps=epsilon)

        # Fix: Intrinsic vector calculation instead of fixed global Z-axis
        d_pmax = torch.sum((centered - p_max.unsqueeze(1)) ** 2, dim=2)
        idx_far = torch.argmax(d_pmax, dim=1)
        p_far = Canonicalizer._gather_points(centered, idx_far.unsqueeze(1)).squeeze(1)

        v2 = p_far - (torch.sum(p_far * u1, dim=1, keepdim=True) * u1)
        n2 = torch.norm(v2, dim=1, keepdim=True)

        # Collinear Fallback
        a = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=pc.dtype).expand(B, 3)
        b = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=pc.dtype).expand(B, 3)
        use_a = (u1[:, 0].abs() < 0.9).unsqueeze(-1)
        aux = torch.where(use_a, a, b)
        u2_fallback = torch.cross(u1, aux, dim=1)

        v2 = torch.where(n2 > 1e-6, v2, u2_fallback)
        u2 = F.normalize(v2, dim=1, eps=epsilon)
        u3 = torch.cross(u1, u2, dim=1)

        R = torch.stack([u1, u2, u3], dim=-1)
        R = Canonicalizer._enforce_so3(R)
        canonical_pc = torch.bmm(centered, R)
        return Canonicalizer._order(canonical_pc)

    @staticmethod
    def lexicographic_sort(pc):
        centered = Canonicalizer._center(pc)
        return Canonicalizer._order(centered)

    @staticmethod
    def radial_sort(pc):
        B, N, D = pc.shape
        centered = Canonicalizer._center(pc)
        norms = torch.sum(centered ** 2, dim=2)
        perm = torch.argsort(norms, dim=1)
        canonical_pc_ord = torch.gather(centered, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        return canonical_pc_ord, perm

    @staticmethod
    def spherical_coordinate_sort(pc):
        B, N, D = pc.shape
        centered = Canonicalizer._center(pc)
        x, y, z = centered[:, :, 0], centered[:, :, 1], centered[:, :, 2]
        r = torch.sqrt(x * x + y * y + z * z) + 1e-8
        # Fix: Protected acos domain mapping
        theta = torch.acos(torch.clamp(z / r, -1.0 + 1e-7, 1.0 - 1e-7))
        phi = torch.atan2(y, x) + math.pi
        keys = theta * 1e3 + phi
        perm = torch.argsort(keys, dim=1)
        canonical_pc_ord = torch.gather(centered, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        return canonical_pc_ord, perm

    @staticmethod
    def spherical_pca(pc, epsilon=1e-8):
        B, N, D = pc.shape
        centered = Canonicalizer._center(pc)
        norms = torch.norm(centered, dim=2, keepdim=True) + epsilon
        dirs = centered / norms
        cov = torch.bmm(dirs.transpose(1, 2), dirs) / (N - 1)
        _, eigenvecs = torch.linalg.eigh(cov)

        # Fix: stable eigen signs
        eigenvecs = Canonicalizer._fix_eig_signs(eigenvecs)
        eigenvecs = Canonicalizer._enforce_so3(eigenvecs)

        canonical_pc = torch.bmm(centered, eigenvecs)
        return Canonicalizer._order(canonical_pc)

    @staticmethod
    def laplacian_embedding_norm(pc, sigma_kernel=1.0, epsilon=1e-8):
        B, N, D = pc.shape
        device = pc.device
        centered = Canonicalizer._center(pc)
        dist_sq = torch.cdist(centered, centered).pow(2)
        W = torch.exp(-dist_sq / (sigma_kernel ** 2))
        W.diagonal(dim1=-2, dim2=-1).fill_(0)
        D_vec = W.sum(dim=2)
        D_inv_sqrt = torch.rsqrt(D_vec + epsilon)
        Dmat = torch.diag_embed(D_inv_sqrt)
        L = torch.eye(N, device=device).unsqueeze(0) - torch.bmm(Dmat, torch.bmm(W, Dmat))

        _, eigvecs = torch.linalg.eigh(L)
        eigvecs = Canonicalizer._fix_eig_signs(eigvecs)
        phi = eigvecs[:, :, 1:4]

        keys = phi[:, :, 0]
        perm = torch.argsort(keys, dim=1)
        canonical_pc_ord = torch.gather(phi, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
        return canonical_pc_ord, perm

    @staticmethod
    def laplacian_embedding_unnorm(pc, sigma_kernel=1.0, epsilon=1e-8):
        B, N, D = pc.shape
        device = pc.device
        centered = Canonicalizer._center(pc)
        dist_sq = torch.cdist(centered, centered).pow(2)
        W = torch.exp(-dist_sq / (sigma_kernel ** 2))
        W.diagonal(dim1=-2, dim2=-1).fill_(0)
        D_vec = W.sum(dim=2)
        L = torch.diag_embed(D_vec) - W

        _, eigvecs = torch.linalg.eigh(L)
        eigvecs = Canonicalizer._fix_eig_signs(eigvecs)
        phi = eigvecs[:, :, 1:4]

        keys = phi[:, :, 0]
        perm = torch.argsort(keys, dim=1)
        canonical_pc_ord = torch.gather(phi, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
        return canonical_pc_ord, perm

    @staticmethod
    def heat_kernel_signature(pc, sigma_kernel=1.0, t=1.0, epsilon=1e-8):
        B, N, D = pc.shape
        device = pc.device
        centered = Canonicalizer._center(pc)
        dist_sq = torch.cdist(centered, centered).pow(2)
        W = torch.exp(-dist_sq / (sigma_kernel ** 2))
        W.diagonal(dim1=-2, dim2=-1).fill_(0)
        D_vec = W.sum(dim=2)
        Dmat = torch.diag_embed(torch.rsqrt(D_vec + epsilon))
        L = torch.eye(N, device=device).unsqueeze(0) - torch.bmm(Dmat, torch.bmm(W, Dmat))

        eigvals, eigvecs = torch.linalg.eigh(L)
        eigvecs = Canonicalizer._fix_eig_signs(eigvecs)
        lam = eigvals[:, 1:4]
        phi = eigvecs[:, :, 1:4]

        exp_terms = torch.exp(-lam.unsqueeze(1) * t)
        # Fix: Correct scalar accumulation by dropping .unsqueeze(-1)
        hks = torch.sum(phi ** 2 * exp_terms, dim=2)
        perm = torch.argsort(hks, dim=1)
        canonical_pc_ord = torch.gather(centered, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        return canonical_pc_ord, perm

    # ==========================
    # SET B: ADVANCED SO(3)
    # ==========================
    @staticmethod
    def extrema_tripod(pc, eps=1e-8):
        centered = Canonicalizer._center(pc)
        B, N, _ = centered.shape
        r2 = (centered ** 2).sum(dim=-1)
        i0 = torch.argmax(r2, dim=1)
        p0 = Canonicalizer._gather_points(centered, i0.view(B, 1)).squeeze(1)
        d0 = ((centered - p0.unsqueeze(1)) ** 2).sum(dim=-1)
        i1 = torch.argmax(d0, dim=1)
        p1 = Canonicalizer._gather_points(centered, i1.view(B, 1)).squeeze(1)
        d1 = ((centered - p1.unsqueeze(1)) ** 2).sum(dim=-1)
        i2 = torch.argmax(d1, dim=1)
        p2 = Canonicalizer._gather_points(centered, i2.view(B, 1)).squeeze(1)
        axis1 = p2 - p1
        v = centered - p1.unsqueeze(1)
        a = axis1.unsqueeze(1)
        cross = torch.cross(v, a, dim=-1)
        score = (cross ** 2).sum(dim=-1) / ((axis1 ** 2).sum(dim=-1, keepdim=True) + eps)
        i3 = torch.argmax(score, dim=1)
        p3 = Canonicalizer._gather_points(centered, i3.view(B, 1)).squeeze(1)
        axis2 = p3 - p1
        R = Canonicalizer._frame_from_two(axis1, axis2, eps)
        canonical = torch.bmm(centered, R)
        canonical, R = Canonicalizer._skew_sign_fix(canonical, R, eps)
        return Canonicalizer._order(canonical)

    @staticmethod
    def mvee_khachiyan(pc, iters=20, eps=1e-7):
        centered = Canonicalizer._center(pc)
        B, N, d = centered.shape
        device, dtype = centered.device, centered.dtype
        X = centered.transpose(1, 2)
        u = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
        I3 = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).repeat(B, 1, 1)
        for _ in range(iters):
            Xu = X * u.unsqueeze(1)
            V = torch.bmm(Xu, X.transpose(1, 2))
            Vinv = torch.linalg.inv(V + eps * I3)
            T = torch.bmm(Vinv, X)
            M = (X * T).sum(dim=1)
            j = torch.argmax(M, dim=1)
            maxM = torch.gather(M, 1, j.view(B, 1)).squeeze(1)
            alpha = (maxM - d - 1.0) / ((d + 1.0) * (maxM - 1.0) + eps)
            alpha = torch.clamp(alpha, 0.0, 0.5)
            u = u * (1.0 - alpha).unsqueeze(1)
            # Fix: Explicit cast tracking
            u = u + alpha.unsqueeze(1) * F.one_hot(j, N).to(device=device, dtype=dtype)

        Xu = X * u.unsqueeze(1)
        V = torch.bmm(Xu, X.transpose(1, 2))
        A = (1.0 / d) * torch.linalg.inv(V + eps * I3)
        vals, vecs = torch.linalg.eigh(A)
        R = vecs.flip(dims=[2])
        R = Canonicalizer._enforce_so3(R)
        canonical = torch.bmm(centered, R)
        canonical, R = Canonicalizer._skew_sign_fix(canonical, R, eps)
        return Canonicalizer._order(canonical)

    @staticmethod
    def fastica_tanh(pc, iters=15, eps=1e-8):
        centered = Canonicalizer._center(pc)
        B, N, _ = centered.shape
        device, dtype = centered.device, centered.dtype
        x = centered / (centered.std(dim=1, keepdim=True) + eps)
        W = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).repeat(B, 1, 1)

        for _ in range(iters):
            Y = torch.bmm(x, W)
            G = torch.tanh(Y)
            Gp = 1.0 - G * G
            W1 = torch.bmm(x.transpose(1, 2), G) / N
            m = Gp.mean(dim=1)
            W1 = W1 - W * m.unsqueeze(1)

            U_w, _, Vh_w = torch.linalg.svd(W1)
            W = torch.bmm(U_w, Vh_w)

        W = Canonicalizer._enforce_so3(W)
        canonical = torch.bmm(centered, W)
        canonical, W = Canonicalizer._skew_sign_fix(canonical, W, eps)
        return Canonicalizer._order(canonical)

    @staticmethod
    def gmm_em_3means(pc, iters=10, eps=1e-6):
        centered = Canonicalizer._center(pc)
        B, N, _ = centered.shape
        device, dtype = centered.device, centered.dtype
        r2 = (centered ** 2).sum(dim=-1)
        perm_r = torch.argsort(r2, dim=1)
        idxs = torch.tensor([0, N // 2, N - 1], device=device).view(1, 3).repeat(B, 1)
        init_idx = torch.gather(perm_r, 1, idxs)
        mu = Canonicalizer._gather_points(centered, init_idx)
        K = 3
        pi = torch.full((B, K), 1.0 / K, device=device, dtype=dtype)
        var = r2.mean(dim=1, keepdim=True).repeat(1, K) / 3.0 + eps
        for _ in range(iters):
            diff = centered.unsqueeze(2) - mu.unsqueeze(1)
            dist_sq = (diff * diff).sum(dim=-1)
            logp = -0.5 * dist_sq / (var.unsqueeze(1) + eps)
            logp = logp - 1.5 * torch.log(var.unsqueeze(1) + eps) + torch.log(pi.unsqueeze(1) + eps)
            resp = torch.softmax(logp, dim=2)
            Nk = resp.sum(dim=1) + eps
            mu = (resp.unsqueeze(-1) * centered.unsqueeze(2)).sum(dim=1) / Nk.unsqueeze(-1)
            var = (resp * dist_sq).sum(dim=1) / (3.0 * Nk) + eps
            pi = Nk / N
        key = mu[:, :, 0] + 1e-3 * mu[:, :, 1] + 1e-6 * mu[:, :, 2]
        comp_perm = torch.argsort(key, dim=1)
        mu = torch.gather(mu, 1, comp_perm.unsqueeze(-1).expand(-1, -1, 3))
        a = mu[:, 2] - mu[:, 0]
        b = mu[:, 1] - 0.5 * (mu[:, 0] + mu[:, 2])
        R = Canonicalizer._frame_from_two(a, b, eps)
        canonical = torch.bmm(centered, R)
        canonical, R = Canonicalizer._skew_sign_fix(canonical, R, eps)
        return Canonicalizer._order(canonical)

    @staticmethod
    def skewness_tensor_power(pc, iters=8, candidates=64, eps=1e-8):
        centered = Canonicalizer._center(pc)
        B, N, _ = centered.shape
        device = centered.device
        r2 = (centered ** 2).sum(dim=-1)
        top = torch.topk(r2, k=min(candidates, N), dim=1, largest=True).indices
        dirs = Canonicalizer._safe_normalize(Canonicalizer._gather_points(centered, top), eps)
        proj = (centered.unsqueeze(2) * dirs.unsqueeze(1)).sum(dim=-1)
        skew = (proj ** 3).mean(dim=1)
        i = torch.argmax(skew.abs(), dim=1)
        w1 = torch.gather(dirs, 1, i.view(B, 1, 1).expand(-1, -1, 3)).squeeze(1)
        for _ in range(iters):
            p = (centered * w1.unsqueeze(1)).sum(dim=-1)
            upd = (centered * (p * p).unsqueeze(-1)).mean(dim=1)
            w1 = Canonicalizer._safe_normalize(upd, eps)
        p1 = (centered * w1.unsqueeze(1)).sum(dim=-1, keepdim=True)
        X2 = centered - p1 * w1.unsqueeze(1)
        proj2 = (X2.unsqueeze(2) * dirs.unsqueeze(1)).sum(dim=-1)
        skew2 = (proj2 ** 3).mean(dim=1)
        i2 = torch.argmax(skew2.abs(), dim=1)
        w2 = torch.gather(dirs, 1, i2.view(B, 1, 1).expand(-1, -1, 3)).squeeze(1)
        w2 = w2 - (w2 * w1).sum(dim=-1, keepdim=True) * w1
        w2 = Canonicalizer._safe_normalize(w2, eps)
        for _ in range(iters):
            p = (X2 * w2.unsqueeze(1)).sum(dim=-1)
            upd = (X2 * (p * p).unsqueeze(-1)).mean(dim=1)
            w2 = Canonicalizer._safe_normalize(upd, eps)
            w2 = w2 - (w2 * w1).sum(dim=-1, keepdim=True) * w1
            w2 = Canonicalizer._safe_normalize(w2, eps)
        R = Canonicalizer._frame_from_two(w1, w2, eps)
        canonical = torch.bmm(centered, R)
        canonical, R = Canonicalizer._skew_sign_fix(canonical, R, eps)
        return Canonicalizer._order(canonical)

    @staticmethod
    def spherical_quadrupole(pc, eps=1e-8):
        centered = Canonicalizer._center(pc)
        r = centered.norm(dim=-1, keepdim=True)
        u = centered / (r + eps)
        w = (r.squeeze(-1) ** 2)
        Uw = u * torch.sqrt(w.unsqueeze(-1) + eps)
        S = torch.bmm(Uw.transpose(1, 2), Uw) / (w.sum(dim=1, keepdim=True).unsqueeze(-1) + eps)
        vals, vecs = torch.linalg.eigh(S)
        R = vecs.flip(dims=[2])
        R = Canonicalizer._enforce_so3(R)
        canonical = torch.bmm(centered, R)
        canonical, R = Canonicalizer._skew_sign_fix(canonical, R, eps)
        return Canonicalizer._order(canonical)

    @staticmethod
    def diffusion_nystrom_frame(pc, m=32, sigma=0.5, eps=1e-8):
        centered = Canonicalizer._center(pc)
        B, N, _ = centered.shape
        device, dtype = centered.device, centered.dtype
        r2 = (centered ** 2).sum(dim=-1)
        perm_r = torch.argsort(r2, dim=1)
        m = min(m, N)
        grid = torch.linspace(0, N - 1, steps=m, device=device).long()
        lm_idx = torch.gather(perm_r, 1, grid.view(1, m).repeat(B, 1))
        L = Canonicalizer._gather_points(centered, lm_idx)
        x2 = (centered ** 2).sum(dim=-1, keepdim=True)
        l2 = (L ** 2).sum(dim=-1).unsqueeze(1)
        xl = torch.bmm(centered, L.transpose(1, 2))
        dist_sq = x2 + l2 - 2.0 * xl
        K = torch.exp(-dist_sq / (sigma * sigma + eps))
        K = K / (K.sum(dim=-1, keepdim=True) + eps)
        A = torch.bmm(K.transpose(1, 2), K) + eps * torch.eye(m, device=device, dtype=dtype).view(1, m, m)
        evals, evecs = torch.linalg.eigh(A)
        V = evecs[:, :, -3:]
        lam = evals[:, -3:]
        Phi = torch.bmm(K, V) / torch.sqrt(lam.unsqueeze(1) + eps)
        i_pos, i_neg = torch.argmax(Phi[:, :, 0], dim=1), torch.argmin(Phi[:, :, 0], dim=1)
        p_pos = Canonicalizer._gather_points(centered, i_pos.view(B, 1)).squeeze(1)
        p_neg = Canonicalizer._gather_points(centered, i_neg.view(B, 1)).squeeze(1)
        u1 = p_pos - p_neg
        j_pos, j_neg = torch.argmax(Phi[:, :, 1], dim=1), torch.argmin(Phi[:, :, 1], dim=1)
        q_pos = Canonicalizer._gather_points(centered, j_pos.view(B, 1)).squeeze(1)
        q_neg = Canonicalizer._gather_points(centered, j_neg.view(B, 1)).squeeze(1)
        u2 = q_pos - q_neg
        R = Canonicalizer._frame_from_two(u1, u2, eps)
        canonical = torch.bmm(centered, R)
        canonical, R = Canonicalizer._skew_sign_fix(canonical, R, eps)
        return Canonicalizer._order(canonical)

    @staticmethod
    def local_pca_normals_frame(pc, k=16, eps=1e-8):
        centered = Canonicalizer._center(pc)
        B, N, _ = centered.shape
        device = centered.device
        k = min(k, N - 1)
        dist = torch.cdist(centered, centered, p=2)
        nn = torch.topk(dist, k=k + 1, dim=-1, largest=False).indices[:, :, 1:]
        neigh = torch.gather(centered.unsqueeze(1).expand(B, N, N, 3), 2, nn.unsqueeze(-1).expand(-1, -1, -1, 3))
        diff = neigh - centered.unsqueeze(2)
        C = torch.matmul(diff.transpose(-1, -2), diff) / (k + eps)
        vals, vecs = torch.linalg.eigh(C)
        nrm = vecs[:, :, :, 0]
        S = torch.bmm(nrm.transpose(1, 2), nrm) / (N + eps)
        evals, evecs = torch.linalg.eigh(S)
        R = evecs.flip(dims=[2])
        R = Canonicalizer._enforce_so3(R)
        canonical = torch.bmm(centered, R)
        canonical, R = Canonicalizer._skew_sign_fix(canonical, R, eps)
        return Canonicalizer._order(canonical)

    @staticmethod
    def l1_principal_frame(pc, iters=20, eps=1e-8):
        centered = Canonicalizer._center(pc)
        B, N, _ = centered.shape
        r2 = (centered ** 2).sum(dim=-1)
        i0 = torch.argmax(r2, dim=1)
        w1 = Canonicalizer._gather_points(centered, i0.view(B, 1)).squeeze(1)
        w1 = Canonicalizer._safe_normalize(w1, eps)
        for _ in range(iters):
            p = (centered * w1.unsqueeze(1)).sum(dim=-1)
            s = torch.sign(p)
            s = torch.where(s == 0, torch.ones_like(s), s)
            upd = (centered * s.unsqueeze(-1)).mean(dim=1)
            w1 = Canonicalizer._safe_normalize(upd, eps)
        p1 = (centered * w1.unsqueeze(1)).sum(dim=-1, keepdim=True)
        X2 = centered - p1 * w1.unsqueeze(1)
        r2_2 = (X2 ** 2).sum(dim=-1)
        i1 = torch.argmax(r2_2, dim=1)
        w2 = Canonicalizer._gather_points(X2, i1.view(B, 1)).squeeze(1)
        w2 = Canonicalizer._safe_normalize(w2, eps)
        for _ in range(iters):
            p = (X2 * w2.unsqueeze(1)).sum(dim=-1)
            s = torch.sign(p)
            s = torch.where(s == 0, torch.ones_like(s), s)
            upd = (X2 * s.unsqueeze(-1)).mean(dim=1)
            w2 = Canonicalizer._safe_normalize(upd, eps)
        R = Canonicalizer._frame_from_two(w1, w2, eps)
        canonical = torch.bmm(centered, R)
        canonical, R = Canonicalizer._skew_sign_fix(canonical, R, eps)
        return Canonicalizer._order(canonical)

    @staticmethod
    def projection_pursuit_kurtosis(pc, candidates=64, eps=1e-8):
        centered = Canonicalizer._center(pc)
        B, N, _ = centered.shape
        r2 = (centered ** 2).sum(dim=-1)
        C = min(candidates, N)
        top = torch.topk(r2, k=C, dim=1, largest=True).indices
        dirs = Canonicalizer._safe_normalize(Canonicalizer._gather_points(centered, top), eps)
        proj = (centered.unsqueeze(2) * dirs.unsqueeze(1)).sum(dim=-1)
        m2 = (proj ** 2).mean(dim=1) + eps
        m4 = (proj ** 4).mean(dim=1)
        kurt = m4 / (m2 * m2) - 3.0
        i = torch.argmax(kurt.abs(), dim=1)
        w1 = torch.gather(dirs, 1, i.view(B, 1, 1).expand(-1, -1, 3)).squeeze(1)
        p1 = (centered * w1.unsqueeze(1)).sum(dim=-1, keepdim=True)
        X2 = centered - p1 * w1.unsqueeze(1)
        proj2 = (X2.unsqueeze(2) * dirs.unsqueeze(1)).sum(dim=-1)
        m2_2 = (proj2 ** 2).mean(dim=1) + eps
        m4_2 = (proj2 ** 4).mean(dim=1)
        kurt2 = m4_2 / (m2_2 * m2_2) - 3.0
        j = torch.argmax(kurt2.abs(), dim=1)
        w2 = torch.gather(dirs, 1, j.view(B, 1, 1).expand(-1, -1, 3)).squeeze(1)
        w2 = w2 - (w2 * w1).sum(dim=-1, keepdim=True) * w1
        w2 = Canonicalizer._safe_normalize(w2, eps)
        R = Canonicalizer._frame_from_two(w1, w2, eps)
        canonical = torch.bmm(centered, R)
        canonical, R = Canonicalizer._skew_sign_fix(canonical, R, eps)
        return Canonicalizer._order(canonical)


# ==========================================
# 3. Evaluation Metrics
# ==========================================

def kabsch_rmsd(X, Y):
    """
    Computes the optimal rigid alignment (Procrustes) between corresponding point sets
    X and Y. Returns the residual RMS distance and the number of reflection flips.
    Replaces Chamfer distance to isolate actual rigid body alignment failures.
    """
    H = torch.bmm(X.transpose(1, 2), Y)
    U, _, Vt = torch.linalg.svd(H)
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

    detR = torch.linalg.det(R)
    neg = (detR < 0)

    # Safely fix reflections
    if neg.any():
        Vt_clone = Vt.clone()
        Vt_clone[neg, -1, :] *= -1
        R_fixed = torch.bmm(Vt_clone.transpose(1, 2), U.transpose(1, 2))
        R = torch.where(neg.view(-1, 1, 1), R_fixed, R)

    Y_aligned = torch.bmm(Y, R)
    err = torch.norm(X - Y_aligned, dim=2).mean(dim=1)
    return err.mean().item(), neg.sum().item()


def spearman_rank_correlation(perm1, perm2):
    """Measures ordering similarity using Spearman Rank."""
    B, N = perm1.shape
    device = perm1.device
    # Fix: properly initialized dtypes
    rank1 = torch.zeros((B, N), device=device, dtype=torch.float32)
    rank2 = torch.zeros((B, N), device=device, dtype=torch.float32)
    idx = torch.arange(N, device=device, dtype=torch.float32).expand(B, N)
    rank1.scatter_(1, perm1, idx)
    rank2.scatter_(1, perm2, idx)
    d_sq = (rank1 - rank2) ** 2
    rho = 1 - (6 * d_sq.sum(dim=1)) / (N * (N ** 2 - 1))
    return rho.mean().item()


# ==========================================
# 4. Benchmark Runner
# ==========================================

def run_benchmark():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}...\n")

    sizes = [1024]
    noise_levels = [0.0, 0.02, 0.05]
    batch_size = 32

    # Map this directly to the path on your Windows machine
    modelnet_path = r"C:\Users\Benjy\Downloads\modelnet40"

    methods = {
        # ... (Keep your existing methods dictionary exactly the same) ...
        "Old Method": Canonicalizer.old_method,
        "PCA + Skewness": Canonicalizer.pca_skew,
        "Spectral/Fiedler": Canonicalizer.spectral_fiedler,
        "ICA Kurtosis": Canonicalizer.ica_kurtosis,
        "Farthest Pair": Canonicalizer.farthest_pair,
        "Max Norm": Canonicalizer.max_norm,
        "Lexicographic Sort": Canonicalizer.lexicographic_sort,
        "Radial Sort": Canonicalizer.radial_sort,
        "Spherical Coord Sort": Canonicalizer.spherical_coordinate_sort,
        "Spherical PCA": Canonicalizer.spherical_pca,
        "Laplacian Norm": Canonicalizer.laplacian_embedding_norm,
        "Laplacian Unnorm": Canonicalizer.laplacian_embedding_unnorm,
        "Heat Kernel Sig": Canonicalizer.heat_kernel_signature,
        "Extrema Tripod": Canonicalizer.extrema_tripod,
        "MVEE Khachiyan": Canonicalizer.mvee_khachiyan,
        "FastICA Tanh": Canonicalizer.fastica_tanh,
        "GMM EM 3-Means": Canonicalizer.gmm_em_3means,
        "Skewness Power": Canonicalizer.skewness_tensor_power,
        "Spherical Quadrupole": Canonicalizer.spherical_quadrupole,
        "Diffusion Nystrom": Canonicalizer.diffusion_nystrom_frame,
        "Local PCA Normals": Canonicalizer.local_pca_normals_frame,
        "L1 Principal": Canonicalizer.l1_principal_frame,
        "Proj Pursuit Kurt": Canonicalizer.projection_pursuit_kurtosis
    }

    for size in sizes:
        print(f"=== Point Cloud Size: {size} points ===")
        # Swap the provider to use ModelNet40
        dataset = PointCloudProvider(
            dataset_type='modelnet40',
            batch_size=batch_size,
            num_points=size,
            device=device,
            data_dir=modelnet_path
        )
        # ... (Rest of the benchmarking loop remains identical) ...

        for noise in noise_levels:
            original_clouds = dataset.get_batch()
            noisy_transformed_clouds = dataset.apply_transforms(original_clouds, noise)

            print(f"  Noise Level: {noise}")
            print(f"    {'Method Name':<21} | {'Kabsch RMS':<10} | {'Rank Corr':<9} | {'Refls':<5} | {'Time (ms)'}")
            print("-" * 75)

            for name, func in methods.items():
                try:
                    start_t = time.time()
                    can_orig, perm_orig = func(original_clouds)
                    can_noisy, perm_noisy = func(noisy_transformed_clouds)
                    elapsed = (time.time() - start_t) * 1000

                    # Replaced Chamfer with Kabsch to test pure rigid-body pose accuracy
                    shape_error, reflections = kabsch_rmsd(can_orig, can_noisy)

                    if perm_orig.shape != perm_noisy.shape:
                        rank_corr = 0.0
                    else:
                        rank_corr = spearman_rank_correlation(perm_orig, perm_noisy)

                    print(
                        f"    {name:<21} | {shape_error:10.4f} | {rank_corr:9.3f} | {reflections:5d} | {elapsed:7.1f}")
                except Exception as e:
                    print(f"    {name:<21} | Failed: {str(e)}")
        print("\n")


if __name__ == "__main__":
    run_benchmark()