import wandb


def main():
    sweep_config = {
        'name': 'Point_Transformer_PCA_A40_Optimized',
        'program': 'main.py',
        'method': 'bayes',
        'metric': {'name': 'test/best_acc', 'goal': 'maximize'},
        'early_terminate': {
            'type': 'hyperband',
            # Bumped min_iter to 30 to give regularized models time to stabilize
            'min_iter': 30,
            'eta': 2
        },
        'parameters': {
            'exp_name': {'value': 'pt_pca_05'},
            'model': {'value': 'point_transformer'},
            'canon_method': {'value': 'pca'},

            # --- CRITICAL FIX: Give the model time to learn with regularization ---
            'epochs': {'value': 150},

            # --- Optimizer & LR ---
            # Higher batch sizes allow for safely pushing higher learning rates
            'lr': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 5e-3},
            'weight_decay': {'values': [0.01, 0.03, 0.05]},

            # --- Hardware Utilization (Pushing the A40 VRAM) ---
            # If 256 throws an OOM, W&B will just fail that run and adapt.
            'batch_size': {'value': 32},

            # --- Architecture (Locked to known good capacity) ---
            'trans_dim': {'value': 216},
            'trans_depth': {'value': 6},
            'trans_heads': {'value': 6},

            # --- Regularization (Forced Non-Zero) ---
            # Targeting the overfitting/confidence issue explicitly
            'dropout': {'values': [0.2, 0.35, 0.5]},
            'drop_path_rate': {'values': [0.05, 0.1, 0.2]},
            'label_smoothing': {'values': [0.1, 0.2]}
        },
        'command': ['${env}', 'python', '${program}', '${args}']
    }

    print("Initializing W&B Sweep...")

    # Set to your lab's entity and project
    entity = "team_nadav"
    project = "modelnet40-canon"

    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)

    print("\n" + "=" * 60)
    print("✅ SWEEP CREATED SUCCESSFULLY!")
    print(f"👉 Point Transformer Sweep ID: {sweep_id}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()