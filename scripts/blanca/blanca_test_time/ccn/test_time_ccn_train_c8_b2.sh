#!/bin/bash
#SBATCH -n 1
#SBATCH --qos=blanca-ccn
#SBATCH --mem=10G
#SBATCH --time=8:00:00
#SBATCH -c 8

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /pl/active/ccnlab/conda/etc/profile.d/conda.sh
conda activate pytorch_source

export MKL_NUM_THREADS=8 OMP_NUM_THREADS=8

echo "MKL_NUM_THREADS: "
echo $MKL_NUM_THREADS
echo "OMP_NUM_THREADS: "
echo $OMP_NUM_THREADS

--dataset CCN \
--train_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/val/ \
--val_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/val/ \
--test_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/test/ \
--seq_len 8 \
--batch_size 2 \
--num_iters 50 \
--model_type PredNet \
--results_dir ../results/train_results \
--out_data_file blanca_test_time_ccn_train_c8_b2.json \
--record_loss_every 1
