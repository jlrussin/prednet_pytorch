#!/usr/bin/env bash
#SBATCH -p local
#SBATCH -A ecortex
#SBATCH --qos=nonpreemptlong
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

echo "Getting preds from autoencoding PredNet with fc sigmoid tanh lr 0.0001 on CCN dataset"

python get_predicted_images.py \
--dataset CCN \
--test_data_path ../data/ccn_images/train/ \
--seq_len 8 \
--last_only True \
--model_type PredNet \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--bias True \
--FC True \
--load_weights_from ../model_weights/train_prednet_fc_sigmoid_tanh_ccn_lr0p0001_lo.pt \
--results_dir ../results/images/fc_sigmoid_tanh_ccn_lr0p0001_lo \
--out_data_file prednet_fc_sigmoid_tanh_ccn_lr0p0001_lo
