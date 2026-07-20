# Weakly Scene Segmentation Using Efficient Transformer 

[![Paper](https://img.shields.io/badge/Paper-IROS%202024-4b44ce.svg)](https://ieeexplore.ieee.org/document/10802479/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Primary dependencies
- Python 3.9+
- PyTorch 2.0+
- Cuda 11.8

Or install dependencies with conda:
```bash
conda env create -f environment.yml
conda activate lstnet
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
```
```angular2html
# NOTE: The versions of the dependencies listed above are only for reference, 
and please check https://pytorch.org/ for pytorch installation command for your CUDA version.
```

## Compile libraries
Follow the instructins [here](lib/README.md) to compile libraries in the `lib` folder.

## Data preparation
The S3DIS dataset can be downloaded from [here](https://cvg-data.inf.ethz.ch/s3dis/) (4.1G). 
Download the `Stanford3dDataset_v1.2_Aligned_Version.zip` file and unzip it. Then, put the unzipped folder into `S3DIS` 
and run the following command:
```bash
cd LST-Net/data
python data_prepare_s3dis.py
```
After pre-processing, the dataset has the following structure:
```angular2html
S3DIS/
├── Stanford3dDataset_v1.2_Aligned_Version/
│   ├── Area_1
│   ├── Area_2
│   ├── Area_3
│   ├── Area_4
│   ├── Area_5
│   ├── Area_6
├── input_0.040
├── original_ply
└── weak_label_0.001
```

### Training
To train the model on the S3DIS dataset, run
```bash
bash tool/train.sh
```
Modify the argument values in `config/s3dis/s3dis_debug.yaml` (e.g., `data_root`, `epochs`, `batch_size`) as needed.
By default, the log and the trained models (`model_last.pth` and `model_best.pth`) will be saved in
`exp/s3dis/<exp_name>/`.

### Test
To test the trained model on the S3DIS dataset, run
```bash
bash tool/test.sh
```
By default it evaluates `model_best.pth` and writes the results to `exp/s3dis/<exp_name>/result/`.


## Acknowledgement
Our code is built upon the following repositories:
[SQN](https://github.com/QingyongHu/SQN),
[RandLA-Net](https://github.com/QingyongHu/RandLA-Net), and
[point-transformer](https://github.com/POSTECH-CVLab/point-transformer).
We would appreciate their authors.

## Citation
```angular2html
If you found this repository is helpful, please cite:

@inproceedings{huang2024weakly,
  title={Weakly Scene Segmentation Using Efficient Transformer},
  author={Huang, Hao and Yuan, Shuaihang and Wen, Congcong and Hao, Yu and Fang, Yi},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024},
  organization={IEEE},
  doi={10.1109/IROS58592.2024.10802479}
}
```

## License
This repository is released under the [MIT License](LICENSE).
