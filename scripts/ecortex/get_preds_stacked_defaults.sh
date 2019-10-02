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

python get_predicted_images.py \
--dataset CCN \
--test_data_path ../data/ccn_images/train/ \
--seq_len 8 \
--model_type StackedConvLSTM \
--R_stack_sizes 3 48 96 192 \
--R_kernel_sizes 3 3 3 3 \
--FC True \
--local_grad False \
--load_weights_from ../model_weights/train_stacked_defaults.pt \
--results_dir ../results/images/stacked_defaults \
--out_data_file stacked_defaults

for gpu in $gpus
do
echo "Setting fan for " $gpu "back to auto"
nvidia_fancontrol auto $gpu
done
