#!/usr/bin/env bash
#SBATCH -p local
#SBATCH -A ecortex
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

python get_predicted_images.py \
--model_type PredNet \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--out_act relu \
--bias True \
--FC True \
--load_weights_from ../model_weights/prednet_kitti_sigmoid_tanh_fc_mse.pt \
--results_dir ../results/images/sigmoid_tanh_fc_mse \
--out_data_file prednet_sigmoid_tanh_fc_mse
