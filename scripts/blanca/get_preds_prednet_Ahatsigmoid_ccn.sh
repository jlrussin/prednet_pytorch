#!/bin/bash
#SBATCH --qos=blanca-ccn
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --nodelist=bnode0202
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=16

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /pl/active/ccnlab/conda/etc/profile.d/conda.sh
conda activate pytorch_mpi

export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16

echo "MKL_NUM_THREADS: "
echo $MKL_NUM_THREADS
echo "OMP_NUM_THREADS: "
echo $OMP_NUM_THREADS

python get_predicted_images.py \
--dataset CCN \
--test_data_path /pl/active/ccnlab/ccn_images/wwi_emer_imgs_20fg_8tick_rot1/train/ \
--seq_len 8 \
--load_weights_from ../model_weights/mp_train_n10_c16_b1_prednet_Ahatsigmoid_ccn.pt \
--results_dir ../results/images/Ahat_sigmoid_ccn \
--out_data_file mp_n10_c16_b1_prednet_Ahat_sigmoid_ccn
