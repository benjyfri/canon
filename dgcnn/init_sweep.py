import wandb


def main():
    sweep_config = {
        'name': 'Point_Transformer_PCA_Refined',
        'program': 'main.py',
        'method': 'bayes',
        'metric': {'name': 'test/best_acc', 'goal': 'maximize'},
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 20,
            'eta': 2
        },
        'parameters': {
            'exp_name': {'value': 'pt_pca_bayes_sweep'},
            'model': {'value': 'point_transformer'},
            'canon_method': {'value': 'pca'},
            'epochs': {'value': 50},

            # --- Optimizer & LR ---
            # Assuming you add a '--weight_decay' flag to your argparse
            'lr': {'distribution': 'log_uniform_values', 'min': 3e-4, 'max': 3e-3},
            'weight_decay': {'values': [1e-5, 3e-5, 1e-4, 1e-2, 3e-2]},
            'batch_size': {'values': [16, 24, 32]},

            # --- Architecture (Kept < 1.5M params) ---
            # Locked heads to 6 to prevent divisibility crashes.
            # 144/6=24 | 192/6=32 | 216/6=36
            'trans_dim': {'values': [144, 192, 216]},
            'trans_depth': {'values': [3, 4, 6]},
            'trans_heads': {'value': 6},

            # --- Regularization ---
            'dropout': {'values': [0.0, 0.1, 0.2, 0.3]},
            # Assuming you add a '--drop_path_rate' flag to your argparse
            'drop_path_rate': {'values': [0.0, 0.05, 0.1, 0.2]},
            # Assuming you add a '--label_smoothing' flag to your argparse
            'label_smoothing': {'values': [0.0, 0.1]}
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