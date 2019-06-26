#!/bin/bash
#SBATCH --qos=blanca-ccn
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --nodelist=bnode[0202-0207,0211,0216,0221,0224]
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=16

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /pl/active/ccnlab/conda/etc/profile.d/conda.sh
conda activate pytorch_mpi

export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16

echo "MKL_NUM_THREADS: "
echo $MKL_NUM_THREADS
echo "OMP_NUM_THREADS: "
echo $OMP_NUM_THREADS

mpirun -n 4 --map-by node:PE=16 python main.py \
--seed 0 \
--dataset CCN \
--train_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/val/ \
--seq_len 8 \
--batch_size 1 \
--num_iters 20 \
--model_type PredNet \
--results_dir ../results/train_results \
--out_data_file blanca_test_time_ccn_train_n4_c16_b1.json \
--record_loss_every 1
