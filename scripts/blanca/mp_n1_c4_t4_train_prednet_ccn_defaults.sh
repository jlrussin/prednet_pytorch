#!/bin/bash
#SBATCH --qos=blanca-ccn
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /pl/active/ccnlab/conda/etc/profile.d/conda.sh
conda activate pytorch_source

echo "Training PredNet with defaults on CCN dataset"
echo "Nodes: 1"
echo "Tasks per node: 4"
echo "CPUs per task: 4"

python main.py \
--num_processes 4 \
--seed 0 \
--dataset CCN \
--train_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/train/ \
--val_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/val/ \
--test_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/test/ \
--seq_len 8 \
--batch_size 1 \
--num_iters 100 \
--out_data_file train_prednet_ccn_defaults_mp.json \
--checkpoint_path ../model_weights/ccn_defaults_mp.pt \
--record_loss_every 1
