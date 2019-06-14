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

echo "Running default prednet trained on kitti with mse to get predicted images"

python get_predicted_images.py \
--model_type ConvLSTM \
--hidden_channels 256 \
--kernel_size 3 \
--in_channels 3 \
--LSTM_act sigmoid \
--LSTM_c_act tanh \
--out_act relu \
--bias True \
--FC True \
--load_weights_from ../model_weights/convlstm_kitti_256h_mse.pt \
--results_dir ../results/images/convlstm \
--out_data_file convlstm_256h_mse
