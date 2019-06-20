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

echo "Training ConvLSTM with MSE on KITTI dataset"

python train.py \
--model_type ConvLSTM \
--hidden_channels 256 \
--kernel_size 3 \
--in_channels 3 \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--out_act relu \
--bias True \
--FC True \
--loss MSE \
--learning_rate 0.001 \
--lr_steps 1 \
--results_dir ../results/train_results \
--out_data_file convlstm_kitti_256h_mse.json \
--checkpoint_path ../model_weights/convlstm_kitti_256h_mse.pt \
--checkpoint_every 20 \
--record_loss_every 500 \
