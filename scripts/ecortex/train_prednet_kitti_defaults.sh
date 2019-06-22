#!/usr/bin/env bash
#SBATCH -p local
#SBATCH -A ecortex
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

echo "Training PredNet with default hyperparameters on KITTI dataset"

python train.py \
--out_data_file train_prednet_kitti_defaults.json \
--checkpoint_path ../model_weights/kitti_defaults.pt \
--checkpoint_every 20 \
--record_loss_every 500
