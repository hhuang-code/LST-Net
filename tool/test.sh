#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --job-name='slt-s3dis'
#SBATCH --exclusive
#SBATCH -p nvidia

#SBATCH --mail-type=END
#SBATCH --mail-user=hh1811@nyu.edu


source ~/.bashrc
conda activate lstnet

cd ./LST-Net

TEST_CODE=test.py

dataset=s3dis
#dataset=scannet
exp_name=debug_20240730_2230_wo_lowrank_wo_sparse
config_name=debug
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
visual_dir=${exp_dir}/visual
config=config/${dataset}/${dataset}_${config_name}.yaml

mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best
mkdir -p ${visual_dir}/last
mkdir -p ${visual_dir}/best

now=$(date +"%Y%m%d_%H%M%S")
#cp ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}
cp tool/test.sh tool/${TEST_CODE} ${exp_dir}

## for srun debug
#$(which python) -u ${exp_dir}/${TEST_CODE} \
#  --config=${config} \
#  save_folder ${result_dir}/best \
#  model_path ${model_dir}/model_best.pth
##

# for submit sbatch job
$(which python) -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_folder ${result_dir}/best \
  visual_folder ${visual_dir}/best \
  model_path ${model_dir}/model_best.pth \
  2>&1 | tee ${exp_dir}/test_best-$now.log

#$(which python) -u ${exp_dir}/${TEST_CODE} \
#  --config=${config} \
#  save_folder ${result_dir}/last \
#  visual_folder ${visual_dir}/last \
#  model_path ${model_dir}/model_last.pth \
#  2>&1 | tee ${exp_dir}/test_last-$now.log
##