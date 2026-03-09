#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

class Canonicalizer:
    """
    Contract: All canonicalizers must return (canonical_pc, perm), where `perm` is
    strictly the argsort index array used to form `canonical_pc` via `torch.gather`.
    This guarantees accurate 1:1 geometry scatter/unsorting for evaluation.
    """

    @staticmethod
    def _center(pc): return pc - pc.mean(dim=1, keepdim=True)
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

    # Add these inside the Canonicalizer class

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

        centered_fied = (fiedler - torch.mean(fiedler, axis=1).unsqueeze(-1))
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
        centered = Canonicalizer._center(pc)

        perm = Canonicalizer.get_fiedler_permutation(centered, sigma_kernel=sigma_kernel, epsilon=1e-8)
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

        # UPDATE: Return R alongside canonical_pc and perm to support pose embedding
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
        B, N, _ = canonical_pc.shape
        device = canonical_pc.device

        # Start with base identity indices
        perm = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        # True lexicographical sort requires sorting stably from least to most significant key.
        # Primary: X (0), Secondary: Y (1), Tertiary: Z (2).
        for dim_idx in (2, 1, 0):
            vals = torch.gather(canonical_pc[..., dim_idx], 1, perm)
            sort_idx = torch.argsort(vals, dim=1, stable=True)
            perm = torch.gather(perm, 1, sort_idx)

        ordered = torch.gather(canonical_pc, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
        return ordered, perm

    @staticmethod
    def pca_skew(pc, epsilon=1e-8):
        B, N, D = pc.shape
        centered = Canonicalizer._center(pc)
        cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)
        # # --- ADD THIS LINE FOR STABILITY ---
        # # Add a small identity matrix to prevent CUSOLVER errors on singular patches
        # cov += torch.eye(D, device=pc.device).unsqueeze(0) * epsilon
        # # ------------------------------------
        # _, eigenvectors = torch.linalg.eigh(cov)

        # 2. Chunking to avoid cuSOLVER batch limits (~32k)
        CHUNK_SIZE = 4096
        eigvec_list = []

        for i in range(0, B, CHUNK_SIZE):
            chunk = cov[i:i + CHUNK_SIZE]
            try:
                _, chunk_vecs = torch.linalg.eigh(chunk)
            except torch._C._LinAlgError:
                # Fallback to CPU for mathematically degenerate patches
                _, chunk_vecs = torch.linalg.eigh(chunk.cpu())
                chunk_vecs = chunk_vecs.to(pc.device)
            eigvec_list.append(chunk_vecs)

        eigenvectors = torch.cat(eigvec_list, dim=0)

        eigenvectors = eigenvectors.flip(dims=[2])
        eigenvectors = Canonicalizer._fix_eig_signs(eigenvectors)
        eigenvectors = Canonicalizer._enforce_so3(eigenvectors)
        canonical_pc = torch.bmm(centered, eigenvectors)
        skew = (canonical_pc ** 3).mean(dim=1)
        s = torch.sign(skew)
        canonical_pc2, R_final = Canonicalizer._apply_data_signs(canonical_pc, eigenvectors, s)
        ordered, perm = Canonicalizer._order(canonical_pc2)

        # UPDATE: Return the final Rotation matrix as well
        return ordered, perm, R_final

# ---------------------------------------------------------------------------
# Highly Optimized PyTorch Utility Functions
# ---------------------------------------------------------------------------

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert batched rotation matrices to quaternions.
    matrix: (..., 3, 3)
    Returns: (..., 4) in (w, x, y, z) format.
    """
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]

    # Calculate all 4 trace options to avoid singularities
    q0 = 1.0 + m00 + m11 + m22
    q1 = 1.0 + m00 - m11 - m22
    q2 = 1.0 - m00 + m11 - m22
    q3 = 1.0 - m00 - m11 + m22

    q_sq = torch.stack([q0, q1, q2, q3], dim=-1)
    _, max_idx = torch.max(q_sq, dim=-1)

    q = torch.zeros_like(q_sq)

    # Branch 0 (w is largest)
    mask0 = (max_idx == 0)
    s0 = torch.sqrt(torch.clamp(q0[mask0], min=1e-6)) * 2.0
    q[mask0, 0] = 0.25 * s0
    q[mask0, 1] = (matrix[mask0, 2, 1] - matrix[mask0, 1, 2]) / s0
    q[mask0, 2] = (matrix[mask0, 0, 2] - matrix[mask0, 2, 0]) / s0
    q[mask0, 3] = (matrix[mask0, 1, 0] - matrix[mask0, 0, 1]) / s0

    # Branch 1 (x is largest)
    mask1 = (max_idx == 1)
    s1 = torch.sqrt(torch.clamp(q1[mask1], min=1e-6)) * 2.0
    q[mask1, 0] = (matrix[mask1, 2, 1] - matrix[mask1, 1, 2]) / s1
    q[mask1, 1] = 0.25 * s1
    q[mask1, 2] = (matrix[mask1, 0, 1] + matrix[mask1, 1, 0]) / s1
    q[mask1, 3] = (matrix[mask1, 0, 2] + matrix[mask1, 2, 0]) / s1

    # Branch 2 (y is largest)
    mask2 = (max_idx == 2)
    s2 = torch.sqrt(torch.clamp(q2[mask2], min=1e-6)) * 2.0
    q[mask2, 0] = (matrix[mask2, 0, 2] - matrix[mask2, 2, 0]) / s2
    q[mask2, 1] = (matrix[mask2, 0, 1] + matrix[mask2, 1, 0]) / s2
    q[mask2, 2] = 0.25 * s2
    q[mask2, 3] = (matrix[mask2, 1, 2] + matrix[mask2, 2, 1]) / s2

    # Branch 3 (z is largest)
    mask3 = (max_idx == 3)
    s3 = torch.sqrt(torch.clamp(q3[mask3], min=1e-6)) * 2.0
    q[mask3, 0] = (matrix[mask3, 1, 0] - matrix[mask3, 0, 1]) / s3
    q[mask3, 1] = (matrix[mask3, 0, 2] + matrix[mask3, 2, 0]) / s3
    q[mask3, 2] = (matrix[mask3, 1, 2] + matrix[mask3, 2, 1]) / s3
    q[mask3, 3] = 0.25 * s3

    return F.normalize(q, p=2, dim=-1)


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    if npoint >= N:
        idx = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
        if npoint > N:
            pad = idx[:, : (npoint - N)]
            idx = torch.cat([idx, pad], dim=1)
        return idx
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
    return centroids


# ---------------------------------------------------------------------------
# Canonical Patch Convolution Layer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Canonical & Spectral Patch Convolution Layers
# ---------------------------------------------------------------------------

class CanonicalPatchLayer(nn.Module):
    def __init__(self, npoint, k, in_channels, mlp_dims):
        super(CanonicalPatchLayer, self).__init__()
        self.npoint = npoint
        self.k = k

        current_dim = k * 3 if in_channels == 0 else k * (3 + in_channels)

        mlp_layers = []
        for out_dim in mlp_dims:
            mlp_layers.append(nn.Conv1d(current_dim, out_dim, kernel_size=1, bias=False))
            mlp_layers.append(nn.BatchNorm1d(out_dim))
            mlp_layers.append(nn.LeakyReLU(negative_slope=0.2))
            current_dim = out_dim

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, xyz, features, fps_idx):
        B, N, _ = xyz.shape

        # Use the externally provided indices
        new_xyz = index_points(xyz, fps_idx)

        sqrdists = square_distance(new_xyz, xyz)
        _, knn_idx = torch.topk(sqrdists, self.k, dim=-1, largest=False, sorted=False)

        grouped_xyz = index_points(xyz, knn_idx)
        grouped_xyz_centered = grouped_xyz - new_xyz.unsqueeze(2)

        patch_pts = grouped_xyz_centered.view(B * self.npoint, self.k, 3)
        canon_pts, perm, R = Canonicalizer.pca_skew(patch_pts)

        if features is not None:
            grouped_features = index_points(features, knn_idx)
            patch_features = grouped_features.view(B * self.npoint, self.k, -1)
            perm_features = perm.unsqueeze(-1).expand(-1, -1, patch_features.size(-1))
            aligned_features = torch.gather(patch_features, 1, perm_features)
            patch_data = torch.cat((canon_pts, aligned_features), dim=-1)
        else:
            patch_data = canon_pts

        patch_flat = patch_data.view(B, self.npoint, -1).transpose(1, 2).contiguous()
        new_features = self.mlp(patch_flat)

        quat = matrix_to_quaternion(R)
        quat = quat.view(B, self.npoint, 4).transpose(1, 2).contiguous()
        centroid = new_xyz.transpose(1, 2).contiguous()

        pose_7d = torch.cat([centroid, quat], dim=1)
        new_features = torch.cat([new_features, pose_7d], dim=1)

        return new_xyz, new_features.transpose(1, 2)


class SpectralPatchLayer(nn.Module):
    def __init__(self, npoint, k, in_channels, mlp_dims, sigma_kernel=1.0):
        super(SpectralPatchLayer, self).__init__()
        self.npoint = npoint
        self.k = k
        self.sigma_kernel = sigma_kernel

        current_dim = k * 3 if in_channels == 0 else k * (3 + in_channels)

        mlp_layers = []
        for out_dim in mlp_dims:
            mlp_layers.append(nn.Conv1d(current_dim, out_dim, kernel_size=1, bias=False))
            mlp_layers.append(nn.BatchNorm1d(out_dim))
            mlp_layers.append(nn.LeakyReLU(negative_slope=0.2))
            current_dim = out_dim

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, xyz, features, fps_idx):
        B, N, _ = xyz.shape

        # Use the externally provided indices
        new_xyz = index_points(xyz, fps_idx)

        sqrdists = square_distance(new_xyz, xyz)
        _, knn_idx = torch.topk(sqrdists, self.k, dim=-1, largest=False, sorted=False)

        grouped_xyz = index_points(xyz, knn_idx)
        grouped_xyz_centered = grouped_xyz - new_xyz.unsqueeze(2)

        patch_pts = grouped_xyz_centered.view(B * self.npoint, self.k, 3)
        canon_pts, perm, R = Canonicalizer.spectral_fiedler(patch_pts, sigma_kernel=self.sigma_kernel)

        if features is not None:
            grouped_features = index_points(features, knn_idx)
            patch_features = grouped_features.view(B * self.npoint, self.k, -1)
            perm_features = perm.unsqueeze(-1).expand(-1, -1, patch_features.size(-1))
            aligned_features = torch.gather(patch_features, 1, perm_features)
            patch_data = torch.cat((canon_pts, aligned_features), dim=-1)
        else:
            patch_data = canon_pts

        patch_flat = patch_data.view(B, self.npoint, -1).transpose(1, 2).contiguous()
        new_features = self.mlp(patch_flat)

        quat = matrix_to_quaternion(R)
        quat = quat.view(B, self.npoint, 4).transpose(1, 2).contiguous()
        centroid = new_xyz.transpose(1, 2).contiguous()

        pose_7d = torch.cat([centroid, quat], dim=1)
        new_features = torch.cat([new_features, pose_7d], dim=1)

        return new_xyz, new_features.transpose(1, 2)

# ---------------------------------------------------------------------------
# Main Hierarchical Models (Multi-Scale Aggregation)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main Hierarchical Models (Upfront FPS & Index Composition)
# ---------------------------------------------------------------------------

class HierarchicalCanonicalNet(nn.Module):
    def __init__(self,
                 sampling=[512, 128, 32],
                 k=16,
                 patch_mlps=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
                 final_mlp_dims=[512, 256],
                 output_channels=40,
                 dropout=0.5):
        super(HierarchicalCanonicalNet, self).__init__()

        assert len(sampling) == len(patch_mlps), "Sampling steps must match patch_mlp stages."
        self.k = k
        self.sampling = sampling  # Save sampling array for the upfront FPS loop

        self.stages = nn.ModuleList()
        in_channels = 0
        accumulated_dim = 0

        for npoint, mlp_dims in zip(sampling, patch_mlps):
            self.stages.append(CanonicalPatchLayer(npoint, k, in_channels, mlp_dims))
            stage_out_channels = mlp_dims[-1] + 7
            in_channels = stage_out_channels
            accumulated_dim += stage_out_channels

        self.fc_layers = nn.ModuleList()
        last_dim = accumulated_dim * 2

        for d in final_mlp_dims:
            self.fc_layers.append(nn.Linear(last_dim, d, bias=False))
            self.fc_layers.append(nn.BatchNorm1d(d))
            self.fc_layers.append(nn.LeakyReLU(negative_slope=0.2))
            self.fc_layers.append(nn.Dropout(p=dropout))
            last_dim = d

        self.fc_layers.append(nn.Linear(last_dim, output_channels))

    def forward(self, x):
        B = x.shape[0]

        # Global Pre-Canonicalization
        x_trans = x.transpose(1, 2).contiguous()
        xyz, _, _ = Canonicalizer.pca_skew(x_trans)

        # ---------------------------------------------------------
        # 1. UPFRONT KERNEL SCHEDULING (Pre-calculate all FPS indices)
        # ---------------------------------------------------------
        current_xyz = xyz
        stage_fps_indices = []

        for npoint in self.sampling:
            fps_idx = farthest_point_sample(current_xyz, npoint)
            stage_fps_indices.append(fps_idx)
            current_xyz = index_points(current_xyz, fps_idx)

        # ---------------------------------------------------------
        # 2. SEQUENTIAL FEATURE COMPUTATION
        # ---------------------------------------------------------
        features = None
        stage_features = []

        for i, stage in enumerate(self.stages):
            # Pass the pre-calculated index into the layer
            xyz, features = stage(xyz, features, stage_fps_indices[i])
            # Keep features as (B, N, C) so index_points slices the spatial dimension, not channels
            stage_features.append(features)

        # ---------------------------------------------------------
        # 3. INDEX COMPOSITION & FEATURE EXTRACTION
        # ---------------------------------------------------------
        final_features = []
        num_stages = len(self.stages)

        for i in range(num_stages):
            f = stage_features[i]

            # If this isn't the final stage, map its features down to the 32 anchor points
            if i < num_stages - 1:
                idx = stage_fps_indices[i + 1]
                for j in range(i + 2, num_stages):
                    idx = torch.gather(idx, 1, stage_fps_indices[j])

                # Slices out the 32 anchors in a single operation
                f = index_points(f, idx)

            final_features.append(f)

        # ---------------------------------------------------------
        # 4. POINT-WISE MULTI-SCALE POOLING
        # ---------------------------------------------------------
        # Concatenate along the channel dimension for the 32 points: (B, 32, accumulated_dim)
        concat_features = torch.cat(final_features, dim=-1)

        # Prepare for global pooling: (B, accumulated_dim, 32)
        concat_features = concat_features.transpose(1, 2).contiguous()

        x_max = F.adaptive_max_pool1d(concat_features, 1).view(B, -1)
        x_avg = F.adaptive_avg_pool1d(concat_features, 1).view(B, -1)
        x_pool = torch.cat([x_max, x_avg], dim=1)

        out = x_pool
        for layer in self.fc_layers:
            out = layer(out)

        return out


class HierarchicalSpectralNet(nn.Module):
    def __init__(self,
                 sampling=[512, 128, 32],
                 k=16,
                 sigma_kernel=1.0,
                 patch_mlps=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
                 final_mlp_dims=[512, 256],
                 output_channels=40,
                 dropout=0.5):
        super(HierarchicalSpectralNet, self).__init__()

        assert len(sampling) == len(patch_mlps), "Sampling steps must match patch_mlp stages."
        self.k = k
        self.sampling = sampling

        self.stages = nn.ModuleList()
        in_channels = 0
        accumulated_dim = 0

        for npoint, mlp_dims in zip(sampling, patch_mlps):
            self.stages.append(SpectralPatchLayer(npoint, k, in_channels, mlp_dims, sigma_kernel=sigma_kernel))
            stage_out_channels = mlp_dims[-1] + 7
            in_channels = stage_out_channels
            accumulated_dim += stage_out_channels

        self.fc_layers = nn.ModuleList()
        last_dim = accumulated_dim * 2

        for d in final_mlp_dims:
            self.fc_layers.append(nn.Linear(last_dim, d, bias=False))
            self.fc_layers.append(nn.BatchNorm1d(d))
            self.fc_layers.append(nn.LeakyReLU(negative_slope=0.2))
            self.fc_layers.append(nn.Dropout(p=dropout))
            last_dim = d

        self.fc_layers.append(nn.Linear(last_dim, output_channels))

    def forward(self, x):
        B = x.shape[0]

        x_trans = x.transpose(1, 2).contiguous()
        xyz, _, _ = Canonicalizer.spectral_fiedler(x_trans)

        current_xyz = xyz
        stage_fps_indices = []

        for npoint in self.sampling:
            fps_idx = farthest_point_sample(current_xyz, npoint)
            stage_fps_indices.append(fps_idx)
            current_xyz = index_points(current_xyz, fps_idx)

        features = None
        stage_features = []

        for i, stage in enumerate(self.stages):
            xyz, features = stage(xyz, features, stage_fps_indices[i])
            # Keep features as (B, N, C) so index_points slices the spatial dimension, not channels
            stage_features.append(features)

        final_features = []
        num_stages = len(self.stages)

        for i in range(num_stages):
            f = stage_features[i]

            if i < num_stages - 1:
                idx = stage_fps_indices[i + 1]
                for j in range(i + 2, num_stages):
                    idx = torch.gather(idx, 1, stage_fps_indices[j])
                f = index_points(f, idx)

            final_features.append(f)

        concat_features = torch.cat(final_features, dim=-1)
        concat_features = concat_features.transpose(1, 2).contiguous()

        x_max = F.adaptive_max_pool1d(concat_features, 1).view(B, -1)
        x_avg = F.adaptive_avg_pool1d(concat_features, 1).view(B, -1)
        x_pool = torch.cat([x_max, x_avg], dim=1)

        out = x_pool
        for layer in self.fc_layers:
            out = layer(out)

        return out