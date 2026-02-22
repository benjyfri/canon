import wandb


def main():
    sweep_config = {
        'name': 'Heavy_5Stage_Hyperparameter_Sweep',
        'program': 'main.py',
        'method': 'bayes',
        'metric': {'name': 'test/best_acc', 'goal': 'maximize'},

        # Hyperband Early Stopping
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10,
            'eta': 2
        },

        'parameters': {
            'exp_name': {'value': 'heavy_5stage_sweep'},
            'epochs': {'value': 50},
            'batch_size': {'value': 32},
            'k': {'value': 20},

            'model': {
                'values': ['hierarchical_canonical', 'hierarchical_spectral']
            },

            # --- LOCKED 5-STAGE TOPOLOGY ---
            'sampling': {'value': [512, 256, 128, 64, 16]},

            # Locked to the heaviest/widest parameter configuration
            'patch_mlps': {
                'value': "[[64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024], [512, 1024, 2048]]"
            },

            # --- TARGETED HYPERPARAMETERS ---
            'lr': {'distribution': 'log_uniform_values', 'min': 0.0001, 'max': 0.005},

            # W&B has room to scale up regularization for the massive concatenation
            'dropout': {'distribution': 'uniform', 'min': 0.15, 'max': 0.5},

            'sigma_kernel': {'distribution': 'uniform', 'min': 0.5, 'max': 3.0}
        },
        'command': ['${env}', 'python', '${program}', '${args}']
    }

    print("Initializing Heavy 5-Stage Sweep with Hyperband...")
    sweep_id = wandb.sweep(sweep_config, project="modelnet40-canon", entity="team_nadav")

    print("\n" + "=" * 50)
    print(f"✅ HEAVY 5-STAGE SWEEP CREATED SUCCESSFULLY!")
    print(f"Your Sweep ID is: {sweep_id}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()