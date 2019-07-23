#!/usr/bin/env bash
#SBATCH -p local
#SBATCH -A ecortex
#SBATCH --qos=nonpreemptlong
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

echo "Doing layer decoding with PredNet with fc sigmoid tanh with lr 0.0001 on CCN dataset"

python layer_decoding.py \
--aggregate_method none \
--weight_decay 0.0 \
--train_data_path ../data/ccn_images/test/ \
--val_data_path ../data/ccn_images/val/ \
--test_data_path ../data/ccn_images/test/ \
--seq_len 8 \
--batch_size 8 \
--num_iters 50 \
--model_type PredNet \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--bias True \
--FC True \
--load_weights_from ../model_weights/train_prednet_fc_sigmoid_tanh_ccn_lr0p0001.pt \
--learning_rate 0.001 \
--results_dir ../results/layer_decoding/ \
--out_data_file prednet_fc_sigmoid_tanh_lr0p0001_ccn_none.json \
--checkpoint_path ../model_weights/train_prednet_fc_sigmoid_tanh_ccn_lr0p0001 \
--checkpoint_every 2 \
--record_loss_every 2
