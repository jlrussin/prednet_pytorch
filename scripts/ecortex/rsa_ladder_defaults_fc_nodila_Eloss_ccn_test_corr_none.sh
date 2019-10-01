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

python RSA.py \
--aggregate_method none \
--similarity_measure corr \
--test_data_path ../data/ccn_images/test/ \
--seq_len 8 \
--idx_dict_hkl ../data/ccn_images/test_label_idx_dict.hkl \
--model_type LadderNet \
--stack_sizes 3 48 96 192 \
--R_stack_sizes 3 48 96 192 \
--A_kernel_sizes 3 3 3 \
--Ahat_kernel_sizes 3 3 3 3 \
--R_kernel_sizes 3 3 3 3 \
--Ahat_act lrelu \
--use_satlu True \
--satlu_act sigmoid \
--local_grad False \
--conv_dilation 1 \
--use_BN True \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--bias True \
--FC True \
--no_R0 True \
--no_skip0 True \
--load_weights_from ../model_weights/train_ladder_defaults_fc_nodila_Eloss.pt \
--results_dir ../results/rsa/ \
--out_data_file ladder_defaults_fc_nodila_Eloss.hkl

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
