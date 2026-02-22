#!/bin/bash
#SBATCH --job-name=spectral_500
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --qos=normal
#SBATCH --nodelist=dym-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=benjyfri96@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# 1. Navigate to the working directory
cd /home/benjamin.fri/PycharmProjects/canon/dgcnn

# 2. Activate the Conda environment
# Adjust this path if your bashrc handles conda differently in non-interactive shells
source /home/benjamin.fri/miniconda3/etc/profile.d/conda.sh
conda activate canon

# 3. Run the training script
# Note: --use_sgd is omitted because your script defaults to True,
# and passing strings to your current boolean setup is buggy!
python main.py \
    --exp_name="spectral_500_run" \
    --model="hierarchical_spectral" \
    --epochs=500 \
    --batch_size=32 \
    --lr=0.0035167752529200657 \
    --k=20 \
    --dropout=0.21549 \
    --sigma_kernel=1.3326354546897403 \
    --patch_mlps="[[64, 128, 256], [256, 256, 512], [512, 1024, 1024]]" \
    --seed=1