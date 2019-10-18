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
--downsample_size 64 \
--seq_len 8 \
--idx_dict_hkl ../data/ccn_images/test_label_idx_dict.hkl \
--model_type PredNet \
--stack_sizes 3 16 32 64 128 256 \
--R_stack_sizes 3 16 32 64 128 256 \
--A_kernel_sizes 3 3 3 3 3 \
--Ahat_kernel_sizes 3 3 3 3 3 3 \
--R_kernel_sizes 3 3 3 3 3 3 \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--bias True \
--FC True \
--load_weights_from ../model_weights/train_prednet_fc_sigmoid_tanh_ccn_lr0p0001_64x64_6l_wd0p01.pt \
--results_dir ../results/rsa/ \
--out_data_file prednet_fc_sigmoid_tanh_ccn_lr0p0001_64x64_6l_wd0p01.hkl


for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
