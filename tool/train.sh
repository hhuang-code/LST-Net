#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=60GB
#SBATCH --time=48:00:00
#SBATCH --job-name='slt-s3dis'
##SBATCH --job-name='slt-scannet'
##SBATCH --exclusive
#SBATCH -p nvidia

#SBATCH --mail-type=END
#SBATCH --mail-user=hh1811@nyu.edu


source ~/.bashrc
conda activate lstnet


cd ./LST-Net

TRAIN_CODE=train.py
TEST_CODE=test.py
MODEL_CODE=sltrm_slr.py

dataset=s3dis
#dataset=scannet
exp_name=debug_20240730_2230_wo_lowrank_wo_sparse
config_name=debug
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${config_name}.yaml

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best
cp tool/train.sh tool/${TRAIN_CODE} ${config} tool/test.sh tool/${TEST_CODE} model/${MODEL_CODE} ${exp_dir}

now=$(date +"%Y%m%d_%H%M%S")

## for srun debug
#$(which python) ${exp_dir}/${TRAIN_CODE} \
#  --config=${config} \
#  save_path ${exp_dir}

# for submit sbatch job
$(which python) ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  resume ${exp_dir}/model/model_last.pth \
  2>&1 | tee ${exp_dir}/train-$now.log
