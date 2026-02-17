import torch
import torch.nn.functional as F
import math
import random
import numpy as np
import time
import h5py
import glob
import os
import plotly.graph_objects as go

# ==========================================
# 0. Determinism
# ==========================================

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_random_rotations(B, device):
    A = torch.randn(B, 3, 3, device=device)
    Q, R_mat = torch.linalg.qr(A)
    # Safe diagonal sign extraction
    d = torch.sign(torch.diagonal(R_mat, dim1=-2, dim2=-1))
    d = torch.where(d == 0, torch.ones_like(d), d)
    Q = Q * d.unsqueeze(-2)
    det = torch.linalg.det(Q)
    flip_mask = (det < 0)
    Q[flip_mask, :, 0] *= -1
    return Q

# ==========================================
# 1. Plotting & Data Provider
# ==========================================

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
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()

    if pc.ndim == 3:
        if pc.shape[0] == 1:
            pc = pc[0]
        else:
            print(f"Warning: Received batch of {pc.shape[0]} point clouds. Plotting the first one.")
            pc = pc[0]

    if pc.shape[1] != 3:
        raise ValueError(f"Expected point cloud with 3 coordinates per point, got shape {pc.shape}")

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=color,
            opacity=0.8,
            line=dict(width=0)
        )
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(title='X', showbackground=False),
            yaxis=dict(title='Y', showbackground=False),
            zaxis=dict(title='Z', showbackground=False),
            aspectmode='data'
        )
    )
    fig.show()

class PointCloudProvider:
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

        all_data = []
        for f in h5_files:
            with h5py.File(f, 'r') as h5_f:
                all_data.append(h5_f['data'][:])

        all_data = np.concatenate(all_data, axis=0)
        self.modelnet_data = torch.tensor(all_data, dtype=torch.float32)
        self.num_total_shapes = self.modelnet_data.shape[0]
        print(f"Successfully loaded {self.num_total_shapes} shapes from ModelNet40.\n")

    @staticmethod
    def _farthest_point_sample(xyz, npoint):
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

# ==========================================
# 2. Canonicalizers (PCA Variants & Spectral)
# ==========================================

class Canonicalizer:
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
        """Fixes eigenvector solver signs deterministically."""
        B, N, K = vecs.shape
        max_idx = torch.argmax(vecs.abs(), dim=1, keepdim=True)
        max_vals = torch.gather(vecs, 1, max_idx)
        signs = torch.sign(max_vals)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return vecs * signs

    @staticmethod
    def _apply_data_signs(canonical_pc, R, s):
        """Applies data-driven sign flips while strictly preserving SO(3)."""
        s = torch.where(s == 0, torch.ones_like(s), s)
        flips = (s < 0).sum(dim=-1)
        odd = (flips % 2 == 1).view(-1)
        s[odd, 2] *= -1
        canonical_pc2 = canonical_pc * s.unsqueeze(1)
        R2 = Canonicalizer._enforce_so3(R * s.unsqueeze(1))
        return canonical_pc2, R2

    @staticmethod
    def _order(canonical_pc):
        B, N, _ = canonical_pc.shape
        device = canonical_pc.device

        perm = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        for dim_idx in (2, 1, 0):
            vals = torch.gather(canonical_pc[..., dim_idx], 1, perm)
            sort_idx = torch.argsort(vals, dim=1, stable=True)
            perm = torch.gather(perm, 1, sort_idx)

        ordered = torch.gather(canonical_pc, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
        return ordered, perm

    @staticmethod
    def _get_base_pca(pc):
        """Extracts the base SO(3) PCA frame, resolving only solver ambiguity."""
        B, N, D = pc.shape
        centered = Canonicalizer._center(pc)
        cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)
        _, eigenvectors = torch.linalg.eigh(cov)
        eigenvectors = eigenvectors.flip(dims=[2])

        eigenvectors = Canonicalizer._fix_eig_signs(eigenvectors)
        eigenvectors = Canonicalizer._enforce_so3(eigenvectors)

        canonical_pc = torch.bmm(centered, eigenvectors)
        return centered, canonical_pc, eigenvectors

    # --- PCA Variants ---

    @staticmethod
    def pca_max_abs(pc, epsilon=1e-8):
        centered, canonical_pc, R = Canonicalizer._get_base_pca(pc)
        max_idx = torch.argmax(canonical_pc.abs(), dim=1, keepdim=True)
        max_vals = torch.gather(canonical_pc, 1, max_idx).squeeze(1)
        s = torch.sign(max_vals)
        canonical_pc2, _ = Canonicalizer._apply_data_signs(canonical_pc, R, s)
        return Canonicalizer._order(canonical_pc2)

    @staticmethod
    def pca_skew(pc, epsilon=1e-8):
        centered, canonical_pc, R = Canonicalizer._get_base_pca(pc)
        skew = (canonical_pc ** 3).mean(dim=1)
        s = torch.sign(skew)
        canonical_pc2, _ = Canonicalizer._apply_data_signs(canonical_pc, R, s)
        return Canonicalizer._order(canonical_pc2)

    @staticmethod
    def pca_median(pc, epsilon=1e-8):
        centered, canonical_pc, R = Canonicalizer._get_base_pca(pc)
        medians = torch.median(canonical_pc, dim=1).values
        s = torch.sign(medians)
        canonical_pc2, _ = Canonicalizer._apply_data_signs(canonical_pc, R, s)
        return Canonicalizer._order(canonical_pc2)

    @staticmethod
    def pca_signed_square(pc, epsilon=1e-8):
        centered, canonical_pc, R = Canonicalizer._get_base_pca(pc)
        signed_sq = (canonical_pc * canonical_pc.abs()).mean(dim=1)
        s = torch.sign(signed_sq)
        canonical_pc2, _ = Canonicalizer._apply_data_signs(canonical_pc, R, s)
        return Canonicalizer._order(canonical_pc2)

    @staticmethod
    def pca_top_k_mean(pc, epsilon=1e-8, k_ratio=0.05):
        B, N, _ = pc.shape
        k = max(1, int(N * k_ratio))
        centered, canonical_pc, R = Canonicalizer._get_base_pca(pc)

        sorted_pc, _ = torch.sort(canonical_pc, dim=1)
        bottom_k_mean = sorted_pc[:, :k, :].mean(dim=1)
        top_k_mean = sorted_pc[:, -k:, :].mean(dim=1)

        s = torch.sign(top_k_mean.abs() - bottom_k_mean.abs())
        canonical_pc2, _ = Canonicalizer._apply_data_signs(canonical_pc, R, s)
        return Canonicalizer._order(canonical_pc2)

    @staticmethod
    def pca_half_variance(pc, epsilon=1e-8):
        centered, canonical_pc, R = Canonicalizer._get_base_pca(pc)
        pos_mask = (canonical_pc > 0).float()
        neg_mask = (canonical_pc < 0).float()

        pos_sq_sum = (canonical_pc * pos_mask).pow(2).sum(dim=1)
        neg_sq_sum = (canonical_pc * neg_mask).pow(2).sum(dim=1)

        s = torch.sign(pos_sq_sum - neg_sq_sum)
        canonical_pc2, _ = Canonicalizer._apply_data_signs(canonical_pc, R, s)
        return Canonicalizer._order(canonical_pc2)

    # --- Spectral Variants ---

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
        vecs = Canonicalizer._fix_eig_signs(vecs)
        fiedler = vecs[:, :, 1]
        sorted_fiedler, perm = torch.sort(fiedler, dim=1)
        return perm

    @staticmethod
    def spectral_fiedler(pc, sigma_kernel=1.0, epsilon=1e-8):
        B, N, D = pc.shape
        device = pc.device
        centered = Canonicalizer._center(pc)
        perm = Canonicalizer.get_fiedler_permutation(centered)
        pc_ord = torch.gather(centered, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        weights = torch.linspace(-1, 1, N, device=device).view(1, N, 1)
        moment_1 = (pc_ord * weights).sum(dim=1, keepdim=True)
        moment_2 = (pc_ord * (weights ** 2)).sum(dim=1, keepdim=True) + epsilon
        moment_3 = torch.cross(moment_1, moment_2, dim=2)
        u1 = F.normalize(moment_1, dim=2, eps=epsilon)
        u2_proj = moment_2 - (moment_2 * u1).sum(dim=2, keepdim=True) * u1
        u2 = F.normalize(u2_proj, dim=2, eps=epsilon)
        u3 = F.normalize(moment_3, dim=2, eps=epsilon)
        R = torch.stack([u1.squeeze(1), u2.squeeze(1), u3.squeeze(1)], dim=-1)
        R = Canonicalizer._enforce_so3(R)
        canonical_pc = torch.bmm(pc_ord, R)
        return canonical_pc, perm

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

# ==========================================
# 3. Evaluation Metrics
# ==========================================

def kabsch_metrics(X, Y):
    """
    Computes true structural RMSD prior to alignment, then computes optimal rigid alignment
    mapping Y -> X to return Geodesic Angular Error and Initial Reflections.
    X and Y MUST have 1:1 point correspondences (unsorted prior to passing).
    """
    # 1. Structural Distance strictly prior to Kabsch (True RMSD native alignment error)
    sq = ((X - Y) ** 2).sum(dim=2)
    rms_per_batch = torch.sqrt(sq.mean(dim=1))
    shape_error_rms = rms_per_batch.mean().item()

    # 2. Rotational Error mapping Y -> X
    H = torch.bmm(Y.transpose(1, 2), X)
    U, _, Vt = torch.linalg.svd(H)
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

    detR = torch.linalg.det(R)
    neg = (detR < 0)
    initial_reflections = neg.sum().item()

    if neg.any():
        Vt_clone = Vt.clone()
        Vt_clone[neg, -1, :] *= -1
        R_fixed = torch.bmm(Vt_clone.transpose(1, 2), U.transpose(1, 2))
        R = torch.where(neg.view(-1, 1, 1), R_fixed, R)

    # Compute geodesic angle of the fixed/final rotation
    trace = R.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    angle_rad = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = angle_rad * (180.0 / math.pi)

    return shape_error_rms, angle_deg.mean().item(), initial_reflections

def spearman_rank_correlation(perm1, perm2):
    B, N = perm1.shape
    device = perm1.device
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

    sizes = [256,512,1024, 2048]
    noise_levels = [0.0, 0.02, 0.05, 0.1, 0.5]
    batch_size = 5000

    # Ensure this path is correct for your setup
    modelnet_path = r"C:\Users\Benjy\Downloads\modelnet40"

    methods = {
        "PCA + Max Abs": Canonicalizer.pca_max_abs,
        "PCA + Skewness (Base)": Canonicalizer.pca_skew,
        "PCA + Median": Canonicalizer.pca_median,
        "PCA + Signed Square": Canonicalizer.pca_signed_square,
        "PCA + Top-K Mean": Canonicalizer.pca_top_k_mean,
        "PCA + Half Variance": Canonicalizer.pca_half_variance,
        "Spectral Fiedler": Canonicalizer.spectral_fiedler,
        "Laplacian Norm Emb": Canonicalizer.laplacian_embedding_norm,
        "Laplacian Unnorm Emb": Canonicalizer.laplacian_embedding_unnorm
    }

    for size in sizes:
        print(f"=== Point Cloud Size: {size} points ===")
        dataset = PointCloudProvider(
            dataset_type='modelnet40',
            batch_size=batch_size,
            num_points=size,
            device=device,
            data_dir=modelnet_path
        )

        for noise in noise_levels:
            original_clouds = dataset.get_batch()
            noisy_transformed_clouds = dataset.apply_transforms(original_clouds, noise)

            print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(f"  Noise Level: {noise}")
            print(f"    {'Method Name':<22} | {'Kabsch RMS':<10} | {'Rot Err (deg)':<13} | {'Rank Corr':<9} | {'Refls':<5} | {'Time (ms)'}")
            print("-" * 89)

            for name, func in methods.items():
                try:
                    start_t = time.time()
                    can_orig, perm_orig = func(original_clouds)
                    can_noisy, perm_noisy = func(noisy_transformed_clouds)
                    elapsed = (time.time() - start_t) * 1000

                    # UNSORT logic: Restores 1:1 geometry mapping for accurate Pose evaluation
                    unsorted_orig = torch.zeros_like(can_orig)
                    unsorted_orig.scatter_(1, perm_orig.unsqueeze(-1).expand(-1, -1, 3), can_orig)

                    unsorted_noisy = torch.zeros_like(can_noisy)
                    unsorted_noisy.scatter_(1, perm_noisy.unsqueeze(-1).expand(-1, -1, 3), can_noisy)

                    shape_error, rot_err, reflections = kabsch_metrics(unsorted_orig, unsorted_noisy)

                    if perm_orig.shape != perm_noisy.shape:
                        rank_corr = 0.0
                    else:
                        rank_corr = spearman_rank_correlation(perm_orig, perm_noisy)

                    print(
                        f"    {name:<22} | {shape_error:10.4f} | {rot_err:13.3f} | {rank_corr:9.3f} | {reflections:5d} | {elapsed:7.1f}")
                except Exception as e:
                    print(f"    {name:<22} | Failed: {str(e)}")
        print("\n")

if __name__ == "__main__":
    run_benchmark()