#!/bin/bash
#SBATCH --qos=blanca-ccn
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /pl/active/ccnlab/conda/etc/profile.d/conda.sh
conda activate pytorch_source

echo "Training PredNet with defaults on KITTI dataset"
echo "Nodes: 1"
echo "Tasks per node: 4"
echo "CPUs per task: 4"

python main.py \
--num_processes 4 \
--seed 0 \
--batch_size 1 \
--num_iters 100 \
--out_data_file train_prednet_kitti_defaults_mp.json \
--checkpoint_path ../model_weights/kitti_defaults_mp.pt \
--record_loss_every 20
