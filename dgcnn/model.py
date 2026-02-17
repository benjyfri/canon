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
        canonical_pc2, _ = Canonicalizer._apply_data_signs(canonical_pc, eigenvectors, s)
        return Canonicalizer._order(canonical_pc2)


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
        canon_pts, perm = Canonicalizer.pca_skew(patch_pts)

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
        canon_x_trans, _ = Canonicalizer.pca_skew(x_trans)
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