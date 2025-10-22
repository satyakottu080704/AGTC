# AGTC Project - Quick Start Guide

## Overview
This project implements Attention-Guided Low-Rank Tensor Completion (AGTC) for hyperspectral image restoration using PyTorch.

## System Requirements
- **Python**: 3.9 recommended (3.8-3.10 should work)
- **CUDA**: 11.3 (for GPU acceleration)
- **RAM**: 16GB+ recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended

## Quick Start Options

### Option 1: Automated Setup (Recommended)
Run the complete setup and training pipeline:

```bash
# For PaviaU dataset (default)
python setup_and_run.py --dataset PaviaU

# For Landsat dataset
python setup_and_run.py --dataset Landsat
```

### Option 2: Use Pretrained Weights (Fastest)
Skip training and use pretrained weights for testing:

```bash
# Test with pretrained weights
python setup_and_run.py --dataset PaviaU --use-pretrained
```

### Option 3: Test Only
If you already have everything set up:

```bash
# Just run testing
python setup_and_run.py --dataset PaviaU --test-only
```

### Option 4: Manual Step-by-Step

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install PyTorch separately with CUDA support:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy h5py pillow opencv-python tqdm torchinfo scipy
```

#### Step 2: Prepare Training Data

**For PaviaU:**
```bash
cd AGTC-HSI/PaviaU/Data-Preparation
python train_pairs_creator_PaviaU.py
cd ../../..
```

**For Landsat:**
```bash
cd AGTC-HSI/Landsat/Data-Preparation
python train_pairs_creator_Landsat.py
cd ../../..
```

#### Step 3: Train the Model

**For PaviaU:**
```bash
cd AGTC-HSI/PaviaU

# Stage 1: Epochs 1-20
python train.py --data_path=./Data-Preparation/Train_Pairs_PaviaU

# Stage 2: Epochs 21-40
python train.py --data_path=./Data-Preparation/Train_Pairs_PaviaU --resume=./checkpoints/epoch_20.pth --set_lr=1e-6

# Stage 3: Epochs 41-60
python train.py --data_path=./Data-Preparation/Train_Pairs_PaviaU --resume=./checkpoints/epoch_40.pth --set_lr=1e-7

# Stage 4: Epochs 61-80
python train.py --data_path=./Data-Preparation/Train_Pairs_PaviaU --resume=./checkpoints/epoch_60.pth --set_lr=1e-8

# Stage 5: Epochs 81-100
python train.py --data_path=./Data-Preparation/Train_Pairs_PaviaU --resume=./checkpoints/epoch_80.pth --set_lr=1e-9

cd ../..
```

#### Step 4: Test the Model
```bash
cd AGTC-HSI/PaviaU
python test.py
cd ../..
```

## Project Structure
```
AGTC/
├── AGTC-HSI/
│   ├── PaviaU/
│   │   ├── Data-Preparation/
│   │   │   ├── PaviaU.mat (raw data)
│   │   │   ├── train_pairs_creator_PaviaU.py (Python script)
│   │   │   └── Train_Pairs_PaviaU/ (generated training data)
│   │   ├── Weight/
│   │   │   └── AGTC-Pavia.pth (pretrained weights)
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── main_net.py
│   │   └── util.py
│   └── Landsat/
│       ├── Data-Preparation/
│       ├── Weight/
│       └── ...
├── env.yml (conda environment spec)
├── requirements.txt (pip requirements)
├── setup_and_run.py (automated setup script)
└── QUICKSTART.md (this file)
```

## Output
- **PaviaU**: `AGTC-HSI/PaviaU/Pavia-AGTC.npy`
- **Landsat**: `AGTC-HSI/Landsat/Landsat-AGTC.npy`

## Notes
- **Training Time**: ~24-48 hours for full 100 epochs on a modern GPU
- **Pretrained Weights**: Available in `Weight/` folders for immediate testing
- **Learning Rate Schedule**: The model requires manual LR adjustments every 20 epochs for best performance
- **Memory**: Batch size is fixed to 1 due to model architecture

## Troubleshooting

### CUDA Out of Memory
- Reduce the number of workers in train.py: `num_workers=2` or `num_workers=1`
- Ensure no other GPU processes are running

### ImportError
- Verify all dependencies are installed: `pip list`
- Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

### Data Preparation Issues
- Ensure .mat files exist in Data-Preparation folders
- Check scipy version: `pip install scipy`

## Citation
If you use this code, please cite:
```
@ARTICLE{Mai2024,
  author={Mai, Truong Thanh Nhat and Lam, Edmund Y. and Lee, Chul},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Attention-Guided Low-Rank Tensor Completion}, 
  year={2024},
  volume={46},
  number={12},
  pages={9818-9833},
  doi={10.1109/TPAMI.2024.3429498}
}
```
