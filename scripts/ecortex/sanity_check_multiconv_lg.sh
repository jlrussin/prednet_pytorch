#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --nodelist=local02
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4

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

python get_predicted_images.py \
--dataset CCN \
--sanity_check True \
--test_data_path ../data/ccn_images/train/ \
--seq_len 8 \
--model_type MultiConvLSTM \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--bias True \
--FC True \
--local_grad True \
--load_weights_from ../model_weights/train_multiconv_fc_sigmoid_tanh_lg_ccn_lr0p0001_llam1p0.pt \
--results_dir ../results/images/multiconv_lg_sanity_check \
--out_data_file multiconv_lg_sanity_check

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
