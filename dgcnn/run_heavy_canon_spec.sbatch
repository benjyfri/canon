#!/bin/bash
#SBATCH --job-name=heavy_canon_500        
#SBATCH --output=logs/heavy_canon_%j.out  
#SBATCH --error=logs/heavy_canon_%j.err             
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=8             
#SBATCH --gres=gpu:1                  
#SBATCH --nodelist=dym-lab3           # Forces the job onto the blazing fast L40S node
#SBATCH --mem=32G                     
#SBATCH --time=48:00:00               # 48 hours to safely clear 500 heavy epochs
#SBATCH --partition=dym
#SBATCH --account=dym-lab    

# Navigate to the working directory
cd ~/PycharmProjects/canon/dgcnn

# Setup environment
eval "$(conda shell.bash hook)"
conda activate canon 

echo "Starting 500-Epoch Marathon for the Heavy 5-Stage Canonical Model..."
echo "Node: $SLURM_NODELIST"

# Run the training loop with the exact optimal hyperparameters from the sweep
python main.py \
    --exp_name="heavy_5stage_500_spectral" \
    --model="hierarchical_spectral" \
    --epochs=500 \
    --batch_size=32 \
    --k=20 \
    --lr=0.0002 \
    --dropout=0.25 \
    --sampling="[512, 256, 128, 64, 16]" \
    --patch_mlps="[[64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024], [512, 1024, 2048]]" \
    --sigma_kernel=0.8172437663359395 \
    --seed=1

echo "Marathon Complete!"
