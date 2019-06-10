#!/usr/bin/env bash
#SBATCH -p local
#SBATCH -A ecortex
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

echo "Training PredNet with MSE and default hyperparameters on KITTI dataset"

python train.py \
--loss MSE \
--out_data_file train_prednet_kitti_defaults_mse.json \
--checkpoint_path ../model_weights/kitti_defaults_mse.pt \
