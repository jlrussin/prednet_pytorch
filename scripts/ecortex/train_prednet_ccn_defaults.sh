#!/usr/bin/env bash
#SBATCH -p local
#SBATCH -A ecortex
#SBATCH --nodelist=local[01,02]
#SBATCH --qos=nonpreemptlong
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

echo "Training PredNet with default hyperparameters on CCN dataset"

python train.py \
--dataset CCN \
--train_data_path ../data/ccn_images/train/ \
--val_data_path ../data/ccn_images/val/ \
--test_data_path ../data/ccn_images/test/ \
--seq_len 8 \
--batch_size 8 \
--num_iters 50000 \
--model_type PredNet \
--results_dir ../results/train_results \
--out_data_file train_prednet_defaults_ccn.json \
--checkpoint_path ../model_weights/train_prednet_defaults_ccn.pt \
--checkpoint_every 2 \
--record_loss_every 200
