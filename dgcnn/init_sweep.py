import wandb


def main():
    sweep_config = {
        'program': 'main.py',
        'method': 'bayes',
        'metric': {'name': 'test/best_acc', 'goal': 'maximize'},
        'parameters': {
            'model': {'value': 'hierarchical_canonical'},
            'epochs': {'value': 150},
            'batch_size': {'values': [16, 32]},
            'lr': {'distribution': 'log_uniform_values', 'min': 0.0005, 'max': 0.05},
            'dropout': {'distribution': 'uniform', 'min': 0.3, 'max': 0.7},
            'k': {'distribution': 'int_uniform', 'min': 12, 'max': 24}
        },
        'command': ['${env}', 'python', '${program}', '${args}']
    }

    print("Initializing Sweep on W&B Servers...")
    # NOTE: Replace "your-username" with your actual W&B username or team name
    sweep_id = wandb.sweep(sweep_config, project="modelnet40-canon")

    print("\n" + "=" * 50)
    print(f"✅ SWEEP CREATED SUCCESSFULLY!")
    print(f"Your Sweep ID is: {sweep_id}")
    print("=" * 50 + "\n")
    print(f"To start training, run: sbatch run_agent.sbatch {sweep_id}")


if __name__ == "__main__":
    main()