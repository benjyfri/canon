import wandb


def main():
    sweep_config = {
        'name': 'Architecture-Size-Search',  # Makes the sweep easy to find in W&B
        'program': 'main.py',
        'method': 'bayes',
        'metric': {'name': 'test/best_acc', 'goal': 'maximize'},
        'parameters': {
            'exp_name': {'value': 'arch_size_search'},  # Labels your local checkpoint folders
            'model': {'value': 'hierarchical_canonical'},
            'epochs': {'value': 50},
            'batch_size': {'value': 32},
            'k': {'value': 20},

            # The Architectures defined directly in the sweep
            'patch_mlps': {
                'values': [
                    "[[32, 64], [64, 128], [128, 256]]",  # Light
                    "[[64, 64, 128], [128, 128, 256], [256, 512, 1024]]",  # Standard
                    "[[64, 128, 256], [256, 256, 512], [512, 1024, 1024]]",  # Wide
                    "[[64, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 1024]]"  # Deep
                ]
            },

            'lr': {'distribution': 'log_uniform_values', 'min': 0.0005, 'max': 0.05},

            # Capped max dropout at 0.5 to prevent underfitting
            'dropout': {'distribution': 'uniform', 'min': 0.2, 'max': 0.5}
        },
        'command': ['${env}', 'python', '${program}', '${args}']
    }

    print("Initializing Architectural Sweep on W&B Servers...")
    sweep_id = wandb.sweep(sweep_config, project="modelnet40-canon", entity="team_nadav")

    print("\n" + "=" * 50)
    print(f"✅ SWEEP CREATED SUCCESSFULLY!")
    print(f"Your Sweep ID is: {sweep_id}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()