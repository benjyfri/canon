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


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Canonicalizer:
    """
    Contract: All canonicalizers must return (canonical_pc, perm), where `perm` is
    strictly the argsort index array used to form `canonical_pc` via `torch.gather`.
    This guarantees accurate 1:1 geometry scatter/unsorting for evaluation.
    """

    @staticmethod
    def _center(pc): return pc - pc.mean(dim=1, keepdim=True)

    @staticmethod
    def _safe_normalize(v, eps=1e-8): return v / (v.norm(dim=-1, keepdim=True) + eps)

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

        max_radii = (torch.max(torch.linalg.norm(centered, axis=2), axis=1))[0].unsqueeze(-1).unsqueeze(-1)
        normalized_patches = centered / torch.clamp(max_radii, min=epsilon)

        perm = Canonicalizer.get_fiedler_permutation(normalized_patches, sigma_kernel=sigma_kernel, epsilon=1e-8)
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


class CanonicalMLP(nn.Module):
    def __init__(self, args, output_channels=40):
        super(CanonicalMLP, self).__init__()
        self.args = args
        self.k = args.k

        # MLPs on flattened ordered patches using 1D Convolutions
        self.conv1 = nn.Sequential(nn.Conv1d(self.k * 3, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(self.k * (3 + 64), 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(self.k * (3 + 64), 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(self.k * (3 + 128), 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(args.emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def get_flattened_canonical_patch(self, pts, x, is_first_layer=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        x_dims = x.size(1)

        x_trans = x.transpose(2, 1).contiguous()

        # Find nearest neighbors dynamically (if first layer, operates on geometry)
        idx = knn(x, k=self.k)

        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        # Extract 3D geometry patches
        pts_flat = pts.view(batch_size * num_points, 3)
        patch_pts = pts_flat[idx, :].view(batch_size * num_points, self.k, 3)

        # Canonicalize local 3D patch to guarantee stable ordering
        canon_pts, perm, R_final = Canonicalizer.pca_skew(patch_pts)

        if is_first_layer:
            # First layer: just use the canonicalized shape
            patch_feature = canon_pts.view(batch_size, num_points, self.k * 3)
        else:
            # Extract feature patches
            x_flat = x_trans.view(batch_size * num_points, x_dims)
            patch_x = x_flat[idx, :].view(batch_size * num_points, self.k, x_dims)

            # Align features using geometry permutation
            perm_x = perm.unsqueeze(-1).expand(-1, -1, x_dims)
            aligned_x = torch.gather(patch_x, 1, perm_x)

            # Concatenate local geometry + aligned features
            patch_feature = torch.cat((canon_pts, aligned_x), dim=-1)
            patch_feature = patch_feature.view(batch_size, num_points, self.k * (3 + x_dims))

        # Permute to (B, channels, N) for the 1D convolution over points
        patch_feature = patch_feature.permute(0, 2, 1).contiguous()
        return patch_feature

    def forward(self, x):
        B, C, N = x.size()

        # 1. Global Canonicalization
        x_trans = x.transpose(1, 2).contiguous()  # (B, N, 3)
        canon_x_trans, _ , R_final= Canonicalizer.pca_skew(x_trans)
        pts = canon_x_trans  # Save base geometry for extracting local patches
        x = canon_x_trans.transpose(1, 2).contiguous()  # (B, 3, N)

        # 2. Patch Extraction & MLPs
        x_patch1 = self.get_flattened_canonical_patch(pts, x, is_first_layer=True)
        x1 = self.conv1(x_patch1)

        x_patch2 = self.get_flattened_canonical_patch(pts, x1, is_first_layer=False)
        x2 = self.conv2(x_patch2)

        x_patch3 = self.get_flattened_canonical_patch(pts, x2, is_first_layer=False)
        x3 = self.conv3(x_patch3)

        x_patch4 = self.get_flattened_canonical_patch(pts, x3, is_first_layer=False)
        x4 = self.conv4(x_patch4)

        x_concat = torch.cat((x1, x2, x3, x4), dim=1)
        x_out = self.conv5(x_concat)

        # Global pooling
        x_max = F.adaptive_max_pool1d(x_out, 1).view(B, -1)
        x_avg = F.adaptive_avg_pool1d(x_out, 1).view(B, -1)
        x_pool = torch.cat((x_max, x_avg), 1)

        # Standard MLP
        x_fc = F.leaky_relu(self.bn6(self.linear1(x_pool)), negative_slope=0.2)
        x_fc = self.dp1(x_fc)
        x_fc = F.leaky_relu(self.bn7(self.linear2(x_fc)), negative_slope=0.2)
        x_fc = self.dp2(x_fc)
        out = self.linear3(x_fc)

        return out


class HybridDGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(HybridDGCNN, self).__init__()
        self.args = args
        self.k = args.k

        # 1. Modular MLP for PCA-Canonicalized 3D Patches
        # Expects args.patch_mlp_dims to be a list of ints, e.g., [64, 64]
        patch_mlp_dims = getattr(args, 'patch_mlp_dims', [64, 64])

        mlp_layers = []
        in_channels = self.k * 3
        for out_channels in patch_mlp_dims:
            mlp_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False))
            mlp_layers.append(nn.BatchNorm1d(out_channels))
            mlp_layers.append(nn.LeakyReLU(negative_slope=0.2))
            in_channels = out_channels

        self.patch_mlp = nn.Sequential(*mlp_layers)

        # Output dimension of the patch MLP to feed into the rest of DGCNN
        dgcnn_dim1 = patch_mlp_dims[-1]

        # 2. Standard DGCNN architecture for the remaining latent embeddings
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv2 = nn.Sequential(nn.Conv2d(dgcnn_dim1 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        # conv5 concatenates the patch_mlp output + conv2 + conv3 + conv4
        concat_dim = dgcnn_dim1 + 64 + 128 + 256
        self.conv5 = nn.Sequential(nn.Conv1d(concat_dim, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # --- 1. First Layer: Extract, Canonicalize, and Embed Local 3D Patches ---
        # Find nearest neighbors on raw 3D geometry
        idx = knn(x, k=self.k)
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        # Extract 3D patches
        x_trans = x.transpose(2, 1).contiguous()
        x_flat = x_trans.view(batch_size * num_points, 3)
        patch_pts = x_flat[idx, :].view(batch_size * num_points, self.k, 3)

        # Canonicalize local patches (PCA + Skew + Sorting)
        canon_pts, _ , R_final= Canonicalizer.pca_skew(patch_pts)

        # Flatten the K points into a single 3*K vector for the MLP
        # Output shape mapping: (B*N, K, 3) -> (B, N, K*3) -> (B, K*3, N)
        canon_flat = canon_pts.view(batch_size, num_points, self.k * 3).permute(0, 2, 1).contiguous()

        # Process through modular MLP to get patch embeddings
        x1 = self.patch_mlp(canon_flat)

        # --- 2. Rest of the layers: Standard DGCNN on Latent Embeddings ---
        # Instead of 3D coords, EdgeConv now operates on the invariant x1 embeddings
        x_latent = get_graph_feature(x1, k=self.k)
        x_latent = self.conv2(x_latent)
        x2 = x_latent.max(dim=-1, keepdim=False)[0]

        x_latent = get_graph_feature(x2, k=self.k)
        x_latent = self.conv3(x_latent)
        x3 = x_latent.max(dim=-1, keepdim=False)[0]

        x_latent = get_graph_feature(x3, k=self.k)
        x_latent = self.conv4(x_latent)
        x4 = x_latent.max(dim=-1, keepdim=False)[0]

        # Concatenate x1 (patch embeddings) + x2 + x3 + x4
        x_concat = torch.cat((x1, x2, x3, x4), dim=1)
        x_features = self.conv5(x_concat)

        # Global Pooling
        x_max = F.adaptive_max_pool1d(x_features, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x_features, 1).view(batch_size, -1)
        x_pool = torch.cat((x_max, x_avg), 1)

        # Final Classification Head
        out = F.leaky_relu(self.bn6(self.linear1(x_pool)), negative_slope=0.2)
        out = self.dp1(out)
        out = F.leaky_relu(self.bn7(self.linear2(out)), negative_slope=0.2)
        out = self.dp2(out)
        out = self.linear3(out)

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F


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

class CanonicalPatchLayer(nn.Module):
    def __init__(self, npoint, k, in_channels, mlp_dims):
        super(CanonicalPatchLayer, self).__init__()
        self.npoint = npoint
        self.k = k

        # The input channels correctly account for the +7 dimensions
        # tracked and passed down from the previous layer via `in_channels`
        current_dim = k * 3 if in_channels == 0 else k * (3 + in_channels)

        mlp_layers = []
        for out_dim in mlp_dims:
            mlp_layers.append(nn.Conv1d(current_dim, out_dim, kernel_size=1, bias=False))
            mlp_layers.append(nn.BatchNorm1d(out_dim))
            mlp_layers.append(nn.LeakyReLU(negative_slope=0.2))
            current_dim = out_dim

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, xyz, features):
        B, N, _ = xyz.shape

        # 1. FPS Downsampling
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)

        # 2. K-NN Grouping
        sqrdists = square_distance(new_xyz, xyz)
        _, knn_idx = torch.topk(sqrdists, self.k, dim=-1, largest=False, sorted=False)

        grouped_xyz = index_points(xyz, knn_idx)
        grouped_xyz_centered = grouped_xyz - new_xyz.unsqueeze(2)

        # 3. Canonicalization
        patch_pts = grouped_xyz_centered.view(B * self.npoint, self.k, 3)

        # Catch the rotation matrix R
        canon_pts, perm, R = Canonicalizer.pca_skew(patch_pts)  # R shape: (B*npoint, 3, 3)

        # 4. Feature Alignment
        if features is not None:
            grouped_features = index_points(features, knn_idx)
            patch_features = grouped_features.view(B * self.npoint, self.k, -1)
            perm_features = perm.unsqueeze(-1).expand(-1, -1, patch_features.size(-1))
            aligned_features = torch.gather(patch_features, 1, perm_features)
            patch_data = torch.cat((canon_pts, aligned_features), dim=-1)
        else:
            patch_data = canon_pts

        # 5. Flattening and MLP
        patch_flat = patch_data.view(B, self.npoint, -1).transpose(1, 2).contiguous()
        new_features = self.mlp(patch_flat)  # (B, out_C, npoint)

        # ---------------------------------------------------------
        # 6. Pose Embeddings (Centroid + Quaternion)
        # ---------------------------------------------------------

        # Calculate Quaternion from Rotation Matrix
        quat = matrix_to_quaternion(R)  # (B*npoint, 4)
        quat = quat.view(B, self.npoint, 4).transpose(1, 2).contiguous()  # (B, 4, npoint)

        # Prepare the centroid
        centroid = new_xyz.transpose(1, 2).contiguous()  # (B, 3, npoint)

        # Concatenate into a 7D Pose Block
        pose_7d = torch.cat([centroid, quat], dim=1)  # (B, 7, npoint)

        # Concatenate 7D Pose to the Semantic Features
        new_features = torch.cat([new_features, pose_7d], dim=1)  # (B, out_C + 7, npoint)

        return new_xyz, new_features.transpose(1, 2)


# ---------------------------------------------------------------------------
# Main Hierarchical Model
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

        self.stages = nn.ModuleList()
        in_channels = 0

        for npoint, mlp_dims in zip(sampling, patch_mlps):
            self.stages.append(CanonicalPatchLayer(npoint, k, in_channels, mlp_dims))

            # The layer explicitly concatenates 7 channels (3 for centroid, 4 for quaternion)
            # to the end of its output, so we must add 7 for the NEXT layer's input
            in_channels = mlp_dims[-1] + 7

        self.fc_layers = nn.ModuleList()

        # in_channels now natively includes the +7 from the final stage
        last_dim = in_channels * 2

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

        # Catch all three returns even if we discard the global permutation and rotation
        xyz, _, _ = Canonicalizer.pca_skew(x_trans)

        features = None

        for stage in self.stages:
            xyz, features = stage(xyz, features)

        features_trans = features.transpose(1, 2).contiguous()
        x_max = F.adaptive_max_pool1d(features_trans, 1).view(B, -1)
        x_avg = F.adaptive_avg_pool1d(features_trans, 1).view(B, -1)
        x_pool = torch.cat([x_max, x_avg], dim=1)

        out = x_pool
        for layer in self.fc_layers:
            out = layer(out)

        return out



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

    def forward(self, xyz, features):
        B, N, _ = xyz.shape

        # 1. FPS Downsampling
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)

        # 2. K-NN Grouping
        sqrdists = square_distance(new_xyz, xyz)
        _, knn_idx = torch.topk(sqrdists, self.k, dim=-1, largest=False, sorted=False)

        grouped_xyz = index_points(xyz, knn_idx)
        grouped_xyz_centered = grouped_xyz - new_xyz.unsqueeze(2)

        # 3. Spectral Canonicalization
        patch_pts = grouped_xyz_centered.view(B * self.npoint, self.k, 3)

        # Catch the rotation matrix R using the new spectral_fiedler method
        canon_pts, perm, R = Canonicalizer.spectral_fiedler(patch_pts, sigma_kernel=self.sigma_kernel)

        # 4. Feature Alignment
        if features is not None:
            grouped_features = index_points(features, knn_idx)
            patch_features = grouped_features.view(B * self.npoint, self.k, -1)
            perm_features = perm.unsqueeze(-1).expand(-1, -1, patch_features.size(-1))
            aligned_features = torch.gather(patch_features, 1, perm_features)
            patch_data = torch.cat((canon_pts, aligned_features), dim=-1)
        else:
            patch_data = canon_pts

        # 5. Flattening and MLP
        patch_flat = patch_data.view(B, self.npoint, -1).transpose(1, 2).contiguous()
        new_features = self.mlp(patch_flat)

        # 6. Pose Embeddings (Centroid + Quaternion)
        quat = matrix_to_quaternion(R)
        quat = quat.view(B, self.npoint, 4).transpose(1, 2).contiguous()
        centroid = new_xyz.transpose(1, 2).contiguous()

        pose_7d = torch.cat([centroid, quat], dim=1)
        new_features = torch.cat([new_features, pose_7d], dim=1)

        return new_xyz, new_features.transpose(1, 2)

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

        self.stages = nn.ModuleList()
        in_channels = 0

        for npoint, mlp_dims in zip(sampling, patch_mlps):
            # Use SpectralPatchLayer instead of CanonicalPatchLayer
            self.stages.append(SpectralPatchLayer(npoint, k, in_channels, mlp_dims, sigma_kernel=sigma_kernel))
            in_channels = mlp_dims[-1] + 7

        self.fc_layers = nn.ModuleList()
        last_dim = in_channels * 2

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

        # Use spectral_fiedler for the global point cloud canonicalization
        xyz, _, _ = Canonicalizer.spectral_fiedler(x_trans)

        features = None

        for stage in self.stages:
            xyz, features = stage(xyz, features)

        features_trans = features.transpose(1, 2).contiguous()
        x_max = F.adaptive_max_pool1d(features_trans, 1).view(B, -1)
        x_avg = F.adaptive_avg_pool1d(features_trans, 1).view(B, -1)
        x_pool = torch.cat([x_max, x_avg], dim=1)

        out = x_pool
        for layer in self.fc_layers:
            out = layer(out)

        return out