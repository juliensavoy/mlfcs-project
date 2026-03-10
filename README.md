# IWDD: Importance-Weighted Diffusion Distillation for Causal Estimation



## Installation

### Prerequisites
- Python ≥ 3.9  
- PyTorch ≥ 1.13  
- CUDA-compatible GPU

### Getting Started
Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/XinranSong/IWDD.git
cd IWDD
```

Create a virtual environment and activate it:

```bash
conda create -n iwdd python=3.9
conda activate iwdd
```
Install the required dependencies:

```bash
pip install -r requirements.txt
```


### Dataset Preparation

The preprocessing procedure for datasets follows the same pipeline as [DiffPO](https://github.com/yccm/DiffPO). Once the original ACIC 2018, ACIC 2016, and IHDP dataset is downloaded,  run the corresponding preprocessing notebook (e.g., `load_ihdp.ipynb`) to generate causal masks and normalized data. The processed files will be saved under:
```
data_ihdp/
├── ihdp_norm_data/
└── ihdp_mask/
```
The preprocessing scripts for ACIC 2018 and ACIC 2016 follow the same structure.

### Running Experiments

#### Example: Single ACIC 2018 Dataset

You can reproduce IWDD results on a specific ACIC 2018 dataset using the provided configuration file:

```bash
CUDA_VISIBLE_DEVICES=1 python exe_acic.py \
    --config acic2018.yaml \
    --current_id "9333a461d3944d089ef60cdf3b88fd40" \
    --pretrain 1 \
    --train_sid 1
```
#### Example: Running Multiple Datasets

For large-scale experiments across multiple ACIC 2018 datasets, use the shell script `script_acic2018.sh`.
Run the scrip with:
```bash
bash script_acic2018.sh
```

## Citation


```bibtex
@misc{song2025generativeframeworkcausalestimation,
      title={A Generative Framework for Causal Estimation via Importance-Weighted Diffusion Distillation}, 
      author={Xinran Song and Tianyu Chen and Mingyuan Zhou},
      year={2025},
      eprint={2505.11444},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.11444}, 
}
```


## Acknowledgments

This implementation builds upon the [SiD](https://github.com/mingyuanzhou/SiD) for diffusion distillation and the [DiffPO](https://github.com/yccm/DiffPO) pipeline for data preprocessing.  


