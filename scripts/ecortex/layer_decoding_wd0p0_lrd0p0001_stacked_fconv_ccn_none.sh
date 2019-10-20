#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=25G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 3

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n")
for gpu in $gpus
do
echo "Setting fan for" $gpu "to full"
nvidia_fancontrol full $gpu
done

python layer_decoding.py \
--aggregate_method none \
--weight_decay 0.0 \
--train_data_path ../data/ccn_images/train/ \
--val_data_path ../data/ccn_images/val/ \
--test_data_path ../data/ccn_images/test/ \
--seq_len 8 \
--batch_size 8 \
--num_iters 200000 \
--model_type StackedConvLSTM \
--R_stack_sizes 3 48 96 192 \
--R_kernel_sizes 3 3 3 3 \
--FC True \
--local_grad False \
--forward_conv True \
--load_weights_from ../model_weights/train_stacked_fconv_lr0p0001.pt \
--learning_rate 0.0001 \
--results_dir ../results/layer_decoding/ \
--out_data_file wd0p0_lrd0p0001_stacked_fconv_ccn_none.json \
--checkpoint_path ../model_weights/train_wd0p0_lrd0p0001_stacked_fconv_ccn \
--checkpoint_every 2 \
--record_loss_every 200

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
