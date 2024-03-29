#!/usr/bin/env bash
#SBATCH -p local
#SBATCH -A ecortex
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

echo "Training PredNet on KITTI dataset"
echo "MSE loss"
echo "LSTM activation: sigmoid"
echo "LSTM inner activation: tanh"
echo "Fully connected: True"

python train.py \
--model_type PredNet \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--out_act relu \
--bias True \
--FC True \
--loss MSE \
--results_dir ../results/train_results \
--out_data_file prednet_kitti_sigmoid_tanh_fc_mse.json \
--checkpoint_path ../model_weights/prednet_kitti_sigmoid_tanh_fc_mse.pt \
--checkpoint_every 20 \
--record_loss_every 500 \
