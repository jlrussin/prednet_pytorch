#!/bin/bash
#SBATCH --qos=blanca-ccn
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /pl/active/ccnlab/conda/etc/profile.d/conda.sh
conda activate pytorch_source

export MKL_NUM_THREADS=4 OMP_NUM_THREADS=4

echo "MKL_NUM_THREADS: "
echo $MKL_NUM_THREADS
echo "OMP_NUM_THREADS: "
echo $OMP_NUM_THREADS

python main.py \
--num_processes 4 \
--seed 0 \
--dataset CCN \
--train_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/val/ \
--val_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/val/ \
--test_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/test/ \
--seq_len 8 \
--batch_size 1 \
--num_iters 50 \
--model_type PredNet \
--results_dir ../results/train_results \
--out_data_file blanca_test_time_ccn_train_c4_b1.json \
--record_loss_every 1
