import torch
import torch.nn as nn
from model_new import GlobalMLPClassifier, PointTransformerClassifier


def get_random_rotation_matrix(device):
    """Generates a random 3x3 SO(3) rotation matrix using QR decomposition."""
    A = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    d = torch.diagonal(R)
    Q = Q * torch.sign(d)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def run_robustness_checks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running robustness checks on: {device.type.upper()}\n")

    B, N, C = 2, 1024, 3
    num_classes = 40
    pc = torch.randn(B, N, C, device=device)

    # 1. Create a permuted version of the point cloud
    perm = torch.randperm(N, device=device)
    pc_permuted = pc[:, perm, :]

    # 2. Create a rotated version of the point cloud
    rot_matrix = get_random_rotation_matrix(device)
    pc_rotated = torch.matmul(pc, rot_matrix.unsqueeze(0))

    configs = [
        ("Global MLP", GlobalMLPClassifier, {'num_points': N}),
        ("Point Transformer", PointTransformerClassifier, {})
    ]

    for model_name, ModelClass, extra_kwargs in configs:
        for method in ['pca', 'spectral']:
            print(f"--- Testing {model_name} [{method.upper()}] ---")

            try:
                model = ModelClass(num_classes=num_classes, canon_method=method, **extra_kwargs).to(device)

                # ==========================================
                # TEST 1: Backward Pass & Gradient Health
                # ==========================================
                model.train()
                model.zero_grad()
                logits = model(pc)
                loss = logits.sum()
                loss.backward()

                # Check if gradients exist and are free of NaNs
                has_nans = False
                no_grad = False
                for name, param in model.named_parameters():
                    if param.grad is None:
                        no_grad = True
                        print(f"  [✗] NO GRADIENT: {name}")
                    elif torch.isnan(param.grad).any():
                        has_nans = True
                        print(f"  [✗] NaN GRADIENT: {name}")

                if not has_nans and not no_grad:
                    print("  [✓] Backward Pass: Gradients flow correctly.")

                # ==========================================
                # TEST 2: Strict Invariance
                # ==========================================
                model.eval()  # Crucial: disable dropout/batchnorm for equality checks
                with torch.no_grad():
                    logits_base = model(pc)
                    logits_perm = model(pc_permuted)
                    logits_rot = model(pc_rotated)

                # We use atol=1e-4 because eigh and complex rotations accumulate small floating point differences
                perm_invariant = torch.allclose(logits_base, logits_perm, atol=1e-4)
                rot_invariant = torch.allclose(logits_base, logits_rot, atol=1e-4)

                if perm_invariant:
                    print("  [✓] Permutation Invariance: Passed.")
                else:
                    max_diff = torch.max(torch.abs(logits_base - logits_perm))
                    print(f"  [✗] Permutation Invariance: FAILED. Max diff: {max_diff:.6f}")

                if rot_invariant:
                    print("  [✓] Rotation Invariance: Passed.")
                else:
                    max_diff = torch.max(torch.abs(logits_base - logits_rot))
                    print(f"  [✗] Rotation Invariance: FAILED. Max diff: {max_diff:.6f}")

            except Exception as e:
                print(f"  [✗] FAILED EXCEPTION: {str(e)}")

            print()


if __name__ == "__main__":
    run_robustness_checks()