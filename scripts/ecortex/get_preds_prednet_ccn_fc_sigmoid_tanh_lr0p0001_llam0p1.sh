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
--test_data_path ../data/ccn_images/train/ \
--seq_len 8 \
--model_type PredNet \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--bias True \
--FC True \
--send_acts True \
--load_weights_from ../model_weights/train_prednet_fc_sigmoid_tanh_ccn_lr0p0001_llam0p1.pt \
--results_dir ../results/images/fc_sigmoid_tanh_ccn_lr0p0001_llam0p1 \
--out_data_file prednet_fc_sigmoid_tanh_ccn_lr0p0001_llam0p1

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
