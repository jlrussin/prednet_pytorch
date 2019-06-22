#!/bin/bash
#SBATCH -n 1
#SBATCH --qos=blanca-ccn
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH -c 12

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

ml git

source /pl/active/ccnlab/conda/etc/profile.d/conda.sh
conda activate pytorch_source

export MKL_NUM_THREADS=12 OMP_NUM_THREADS=12

echo "MKL_NUM_THREADS: "
echo $MKL_NUM_THREADS
echo "OMP_NUM_THREADS: "
echo $OMP_NUM_THREADS

python train.py \
--train_data_hkl ../data/kitti_data/X_val.hkl \
--train_sources_hkl ../data/kitti_data/sources_val.hkl \
--val_data_hkl ../data/kitti_data/X_val.hkl \
--val_sources_hkl ../data/kitti_data/sources_val.hkl \
--test_data_hkl ../data/kitti_data/X_val.hkl \
--test_sources_hkl ../data/kitti_data/sources_val.hkl \
--batch_size 4 \
--num_iters 20 \
--lr_steps 0 \
--out_data_file time_test.json \
--record_loss_every 1
