#!/bin/bash
#SBATCH --qos=blanca-ccn
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH -c 8

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

ml git

source /pl/active/ccnlab/conda/etc/profile.d/conda.sh
conda activate pytorch_source

echo "Training PredNet with defaults on CCN dataset"

python train.py \
--dataset CCN \
--train_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/train/ \
--val_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/val/ \
--test_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/test/ \
--batch_size 1 \
--model_type PredNet \
--results_dir ../results/train_results \
--out_data_file prednet_ccn_defaults.json \
--checkpoint_path ../model_weights/prednet_ccn_defaults.pt \
--checkpoint_every 20 \
--record_loss_every 100
