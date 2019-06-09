#!/usr/bin/env bash
#SBATCH -p local
#SBATCH -A ecortex
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 1

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

echo "Training PredNet on tiny KITTI dataset"

python train.py \
--train_data_hkl ../data/kitti_data/small_test.hkl \
--train_sources_hkl ../data/kitti_data/sources_small_test.hkl \
--val_data_hkl ../data/kitti_data/small_test.hkl \
--val_sources_hkl ../data/kitti_data/sources_small_test.hkl \
--test_data_hkl ../data/kitti_data/small_test.hkl \
--test_sources_hkl ../data/kitti_data/sources_small_test.hkl \
--seq_len 2 \
--batch_size 2 \
--num_iters 10 \
--model_type PredNet \
--stack_sizes 3 10 10 \
--R_stack_sizes 3 10 10 \
--A_kernel_sizes 3 3 \
--Ahat_kernel_sizes 3 3 3 \
--R_kernel_sizes 3 3 3 \
--use_satlu True \
--satlu_act hardtanh \
--pixel_max 255.0 \
--error_act relu \
--use_1x1_out False \
--in_channels 3 \
--LSTM_act tanh \
--LSTM_c_act hardsigmoid \
--bias True \
--FC False \
--loss E \
--learning_rate 0.001 \
--lr_steps 1 \
--time0_lambda 0.0 \
--layer_lambdas 1.0 0.0 0.0 \
--results_dir results \
--out_data_file small_test_results.json \
--checkpoint_every 5 \
--record_loss_every 50 \
