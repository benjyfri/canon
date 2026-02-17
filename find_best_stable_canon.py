import torch
import torch.nn.functional as F
import math


# ==========================================
# 1. Data Generation & Augmentation
# ==========================================

def generate_synthetic_clouds(batch_size, num_points, device='cpu'):
    """Generates random ellipsoids to simulate anisotropic point clouds."""
    clouds = torch.randn(batch_size, num_points, 3, device=device)
    # Scale axes differently to give clear principal directions
    scales = torch.tensor([1.0, 0.5, 0.2], device=device).view(1, 1, 3)
    clouds = clouds * scales
    return clouds


def apply_random_transform_and_noise(clouds, noise_std=0.01):
    """Applies random rotations, translations, and Gaussian noise."""
    B, N, _ = clouds.shape
    device = clouds.device

    # Random Translation
    translations = torch.randn(B, 1, 3, device=device) * 5.0

    # Random Rotation
    angles = torch.rand(B, 3, device=device) * 2 * math.pi
    rotations = []
    for i in range(B):
        cx, cy = torch.cos(angles[i]), torch.sin(angles[i])
        Rx = torch.tensor([[1, 0, 0], [0, cx[0], -cy[0]], [0, cy[0], cx[0]]], device=device)
        Ry = torch.tensor([[cx[1], 0, cy[1]], [0, 1, 0], [-cy[1], 0, cx[1]]], device=device)
        Rz = torch.tensor([[cx[2], -cy[2], 0], [cy[2], cx[2], 0], [0, 0, 1]], device=device)
        R = Rz @ Ry @ Rx
        rotations.append(R)

    R_batch = torch.stack(rotations)

    # Apply Transform and Noise
    transformed = torch.bmm(clouds, R_batch.transpose(1, 2)) + translations
    noise = torch.randn_like(transformed) * noise_std

    return transformed + noise


# ==========================================
# 2. Canonicalization Methods
# ==========================================

class Canonicalizer:

    @staticmethod
    def old_method(pc, epsilon=1e-10):
        """Your original canonicalization method using the Laplacian and last-point alignment."""
        B, N, D = pc.shape
        device = pc.device

        # 1. Create Laplacian
        distances = torch.cdist(pc, pc)
        weights = torch.exp(-distances ** 2)
        column_sums = weights.sum(dim=1)
        inv_sqrt_col_sum = torch.rsqrt(column_sums + 1e-7)
        laplacian = torch.eye(N, device=device).unsqueeze(0) - (
                    inv_sqrt_col_sum.unsqueeze(2) * weights * inv_sqrt_col_sum.unsqueeze(1))

        # 2. Top K smallest eigenvectors (k=1)
        _, eigenvectors = torch.linalg.eigh(laplacian)
        first_eig_vec = eigenvectors[:, :, 1]  # Index 1 is the first non-zero eigenvector

        # 3. Sort by first eigenvector
        abs_tensor = torch.abs(first_eig_vec)
        max_indices = torch.argmax(abs_tensor, dim=1)
        max_values = first_eig_vec[torch.arange(B), max_indices]
        signs = torch.sign(max_values)
        result = first_eig_vec * signs.unsqueeze(1)

        # We sort all points here (removed the zero-pinning for global evaluation)
        perm = torch.argsort(result, dim=1)

        # Gather data
        data = torch.gather(pc, 1, perm.unsqueeze(2).expand(-1, -1, 3))

        # 4. Transform to canonical (Your original math)
        m_point = torch.mean(data, dim=1)
        m_norm = torch.norm(m_point, dim=1, keepdim=True)
        v1 = m_point / (m_norm + epsilon)
        v2 = torch.zeros_like(m_point)
        v2[:, 2] = 1.0

        rotation_axis = torch.cross(v1, v2, dim=1)
        rotation_axis = rotation_axis / (torch.norm(rotation_axis, dim=1, keepdim=True) + epsilon)
        cos_theta = torch.sum(v1 * v2, dim=1)
        sin_theta = torch.sqrt(torch.clamp(1 - cos_theta ** 2, min=0) + epsilon)

        K = torch.zeros((B, 3, 3), device=device)
        K[:, 0, 1], K[:, 0, 2] = -rotation_axis[:, 2], rotation_axis[:, 1]
        K[:, 1, 0], K[:, 1, 2] = rotation_axis[:, 2], -rotation_axis[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -rotation_axis[:, 1], rotation_axis[:, 0]

        R1 = torch.eye(3, device=device).unsqueeze(0) + \
             sin_theta.unsqueeze(-1).unsqueeze(-1) * K + \
             (1 - cos_theta).unsqueeze(-1).unsqueeze(-1) * torch.bmm(K, K)

        rotated_points = torch.bmm(data, R1.transpose(1, 2))
        rotated_p = rotated_points[:, -1, :]

        xy_magnitude = torch.sqrt(rotated_p[:, 0] ** 2 + rotated_p[:, 1] ** 2 + epsilon)
        cos_phi = rotated_p[:, 0] / xy_magnitude
        sin_phi = -rotated_p[:, 1] / xy_magnitude

        R2 = torch.zeros((B, 3, 3), device=device)
        R2[:, 0, 0], R2[:, 0, 1] = cos_phi, -sin_phi
        R2[:, 1, 0], R2[:, 1, 1] = sin_phi, cos_phi
        R2[:, 2, 2] = 1.0

        final_points = torch.bmm(rotated_points, R2.transpose(1, 2))
        final_points[:, -1, 1] = 0
        final_points = torch.where(torch.abs(final_points) < epsilon, torch.zeros_like(final_points), final_points)

        return final_points, perm

    @staticmethod
    def pca_skew(pc, epsilon=1e-8):
        """PCA + Skewness resolving."""
        B, N, D = pc.shape
        centroid = pc.mean(dim=1, keepdim=True)
        centered = pc - centroid

        cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)
        _, eigenvectors = torch.linalg.eigh(cov)
        eigenvectors = eigenvectors.flip(dims=[2])  # Descending order

        canonical_pc = torch.bmm(centered, eigenvectors)
        skewness = (canonical_pc ** 3).mean(dim=1, keepdim=True)
        sign_flip = torch.sign(skewness)
        sign_flip[sign_flip == 0] = 1.0

        canonical_pc = canonical_pc * sign_flip

        # Order by primary canonical axis (X) to generate a permutation
        # We add slight weighting to Y and Z to prevent exact ties
        sorting_keys = canonical_pc[:, :, 0] + canonical_pc[:, :, 1] * 1e-4 + canonical_pc[:, :, 2] * 1e-6
        perm = torch.argsort(sorting_keys, dim=1)
        canonical_pc_ord = torch.gather(canonical_pc, 1, perm.unsqueeze(-1).expand(-1, -1, D))

        return canonical_pc_ord, perm

    @staticmethod
    def spectral_fiedler(pc, sigma_kernel=0.1, epsilon=1e-8):
        """2D-style Fiedler vector ordering and alignment adapted for 3D."""
        B, N, D = pc.shape
        device = pc.device

        centroid = pc.mean(dim=1, keepdim=True)
        centered = pc - centroid

        dist_sq = torch.cdist(centered, centered, p=2).pow(2)
        W = torch.exp(-dist_sq / (sigma_kernel ** 2))
        mask = torch.eye(N, device=device).bool().unsqueeze(0).expand(B, -1, -1)
        W.masked_fill_(mask, 0)

        D_vec = W.sum(dim=2) + epsilon
        D_inv_sqrt = torch.rsqrt(D_vec)
        W_norm = W * D_inv_sqrt.unsqueeze(1) * D_inv_sqrt.unsqueeze(2)
        L_sym = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1) - W_norm

        vals, vecs = torch.linalg.eigh(L_sym)
        fiedler = vecs[:, :, 1] * D_inv_sqrt

        skew = (fiedler ** 3).sum(dim=1, keepdim=True)
        sign_flip = torch.where(skew >= 0, torch.ones_like(skew), -torch.ones_like(skew))
        fiedler = fiedler * sign_flip

        perm = torch.argsort(fiedler, dim=1)
        pc_ord = torch.gather(centered, 1, perm.unsqueeze(-1).expand(-1, -1, D))

        weights = torch.linspace(-1, 1, N, device=device).view(1, N, 1)
        moment_1 = (pc_ord * weights).sum(dim=1, keepdim=True)
        moment_2 = (pc_ord * (weights ** 2)).sum(dim=1, keepdim=True)
        moment_3 = torch.cross(moment_1, moment_2, dim=2)

        u1 = F.normalize(moment_1, dim=2, eps=epsilon)
        u2_proj = moment_2 - (moment_2 * u1).sum(dim=2, keepdim=True) * u1
        u2 = F.normalize(u2_proj, dim=2, eps=epsilon)
        u3 = F.normalize(moment_3, dim=2, eps=epsilon)

        R = torch.cat([u1, u2, u3], dim=1).transpose(1, 2)
        canonical_pc = torch.bmm(pc_ord, R.transpose(1, 2))

        return canonical_pc, perm


# ==========================================
# 3. Evaluation Metrics
# ==========================================

def chamfer_distance_approx(pc1, pc2):
    """Measures geometric overlap (Shape Error)."""
    dist = torch.cdist(pc1, pc2)
    min_dist_12 = torch.min(dist, dim=2)[0].mean(dim=1)
    min_dist_21 = torch.min(dist, dim=1)[0].mean(dim=1)
    return (min_dist_12 + min_dist_21).mean().item()


def permutation_accuracy(perm1, perm2):
    """Measures exact ordering overlap (% of points matching)."""
    correct = (perm1 == perm2).float()
    return correct.mean().item() * 100.0  # Returns percentage


def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}...\n")

    sizes = [100, 500]
    noise_levels = [0.0, 0.02, 0.05]
    batch_size = 16

    methods = {
        "Your Old Method": Canonicalizer.old_method,
        "PCA + Skewness": Canonicalizer.pca_skew,
        "Spectral/Fiedler": Canonicalizer.spectral_fiedler
    }

    for size in sizes:
        print(f"=== Point Cloud Size: {size} points ===")
        for noise in noise_levels:
            original_clouds = generate_synthetic_clouds(batch_size, size, device)
            noisy_transformed_clouds = apply_random_transform_and_noise(original_clouds, noise)

            print(f"  Noise Level: {noise}")
            for name, func in methods.items():
                try:
                    can_orig, perm_orig = func(original_clouds)
                    can_noisy, perm_noisy = func(noisy_transformed_clouds)

                    shape_error = chamfer_distance_approx(can_orig, can_noisy)
                    order_acc = permutation_accuracy(perm_orig, perm_noisy)

                    print(f"    {name:<18} | Shape Err: {shape_error:.4f} | Ordering Acc: {order_acc:5.1f}%")
                except Exception as e:
                    print(f"    {name:<18} | Failed: {str(e)}")
        print()


if __name__ == "__main__":
    run_benchmark()