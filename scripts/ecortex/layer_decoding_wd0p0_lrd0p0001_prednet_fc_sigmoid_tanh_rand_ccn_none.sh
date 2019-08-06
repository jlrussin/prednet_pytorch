#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=32G
#SBATCH --time=48:00:00
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
echo $gpu
nvidia_fancontrol full $gpu
done

echo "Doing layer decoding with PredNet with fc sigmoid tanh with lr 0.0001 on CCN dataset"
echo "Decoder learning rate: 0.0001"
echo "Decoder weight decay: 0.0"

python layer_decoding.py \
--aggregate_method none \
--weight_decay 0.0 \
--train_data_path ../data/ccn_images/train/ \
--val_data_path ../data/ccn_images/val/ \
--test_data_path ../data/ccn_images/test/ \
--seq_len 8 \
--batch_size 8 \
--num_iters 200000 \
--model_type PredNet \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--bias True \
--FC True \
--learning_rate 0.0001 \
--results_dir ../results/layer_decoding/ \
--out_data_file wd0p0_lrd0p0001_prednet_fc_sigmoid_tanh_rand_ccn_none.json \
--checkpoint_path ../model_weights/train_wd0p0_lrd0p0001_prednet_fc_sigmoid_tanh_ccn_rand \
--checkpoint_every 2 \
--record_loss_every 200

gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n")
for gpu in $gpus
do
echo $gpu
nvidia_fancontrol auto $gpu
done
