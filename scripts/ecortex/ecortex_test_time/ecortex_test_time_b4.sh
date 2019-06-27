#!/usr/bin/env bash
#SBATCH -p local
#SBATCH -A ecortex
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

echo "Training PredNet with default hyperparameters on KITTI dataset"

python train.py \
--train_data_path ../data/kitti_data/X_val.hkl \
--train_sources_path ../data/kitti_data/sources_val.hkl \
--val_data_path ../data/kitti_data/X_val.hkl \
--val_sources_path ../data/kitti_data/sources_val.hkl \
--test_data_path ../data/kitti_data/X_val.hkl \
--test_sources_path ../data/kitti_data/sources_val.hkl \
--batch_size 4 \
--num_iters 20 \
--lr_steps 0 \
--out_data_file time_test.json \
--record_loss_every 1
