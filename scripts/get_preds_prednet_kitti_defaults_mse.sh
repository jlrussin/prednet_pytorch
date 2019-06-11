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
--load_weights_from ../model_weights/kitti_defaults_mse.pt \
--results_dir ../results/images/defaults_mse \
--out_data_file prednet_defaults_mse
