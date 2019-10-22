#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=25G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 3

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n")
for gpu in $gpus
do
echo "Setting fan for" $gpu "to full"
nvidia_fancontrol full $gpu
done

python train.py \
--dataset CCN \
--train_data_path ../data/ccn_images/train/ \
--val_data_path ../data/ccn_images/val/ \
--test_data_path ../data/ccn_images/test/ \
--seq_len 8 \
--batch_size 8 \
--num_iters 150000 \
--model_type PredNet \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--bias True \
--FC True \
--send_acts False \
--no_A_conv True \
--local_grad False \
--stack_sizes 3 48 96 \
--R_stack_sizes 3 48 96 \
--A_kernel_sizes 3 3 \
--Ahat_kernel_sizes 3 3 3 \
--R_kernel_sizes 3 3 3 \
--layer_lambdas 1.0 0.0 0.0 \
--learning_rate 0.0001 \
--lr_steps 0 \
--results_dir ../results/train_results \
--out_data_file train_prednet_fc_sigmoid_tanh_noconv_3l_ccn_lr0p0001.json \
--checkpoint_path ../model_weights/train_prednet_fc_sigmoid_tanh_noconv_3l_ccn_lr0p0001.pt \
--checkpoint_every 2 \
--record_E True \
--record_loss_every 200

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
