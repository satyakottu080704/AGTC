# Landsat Hyperspectral Image Restoration - Usage Guide

## Overview
This directory contains a complete pipeline for hyperspectral image restoration on Landsat data using the AGTC (Adaptive Graph Tensor Completion) model.

## Project Structure
```
Landsat/
├── Data-Preparation/           # Data preprocessing scripts
│   └── train_pairs_creator_Landsat.py
├── Weight/                     # Saved model weights
├── Metrics/                    # Training and testing metrics
├── main_net.py                 # RPCA-Net model architecture
├── train.py                    # Training script
├── test.py                     # Testing script
├── util.py                     # Utility functions
├── run_landsat_pipeline.py     # Main executable pipeline
└── Landsat_test.mat            # Test data

## Requirements
- Python 3.7+
- PyTorch with CUDA support (recommended)
- Required packages: numpy, scipy, h5py, tqdm, torchinfo, scikit-image

Install dependencies:
```bash
pip install torch torchvision numpy scipy h5py tqdm torchinfo scikit-image
```

## Quick Start

### Option 1: Run Complete Pipeline
Execute the entire pipeline (data preparation → training → testing):
```bash
python run_landsat_pipeline.py --mode all
```

### Option 2: Run Individual Steps

#### Step 1: Data Preparation
Prepare training pairs from the source data in "New folder":
```bash
python run_landsat_pipeline.py --mode prepare
```
Or directly:
```bash
cd Data-Preparation
python train_pairs_creator_Landsat.py
cd ..
```

#### Step 2: Training
Train the model with prepared data:
```bash
python run_landsat_pipeline.py --mode train
```
Or directly:
```bash
python train.py --data_path ./Data-Preparation/Train_Pairs_Landsat --N_iter 10 --input_dim 8
```

**Training Parameters:**
- `--data_path`: Path to training data (required)
- `--save_path`: Path to save weights (default: ./Weight)
- `--metrics_path`: Path to save metrics (default: ./Metrics)
- `--N_iter`: Number of unrolled iterations (default: 10)
- `--input_dim`: Number of spectral bands (default: 8 for Landsat)
- `--checkpoint_freq`: Save checkpoint every N epochs (default: 2)
- `--resume`: Resume from checkpoint
- `--set_lr`: Set custom learning rate

#### Step 3: Testing
Test the trained model:
```bash
python run_landsat_pipeline.py --mode test
```
Or directly:
```bash
python test.py --ckpt_path ./Weight/AGTC-Landsat.pth --N_iter 10 --input_dim 8
```

**Testing Parameters:**
- `--ckpt_path`: Path to model checkpoint
- `--N_iter`: Number of iterations (must match training)
- `--input_dim`: Number of spectral bands (must match training)
- `--test_data`: Path to test data (default: Landsat_test.mat)
- `--ground_truth`: Path to ground truth for metrics (optional)

## Data Format

### Input Data (from "New folder/landsat/")
- `Landsat7_training_clean.npy`: Clean hyperspectral images (shape: H×W×8)
- `Landsat7_training_mask.npy`: Corresponding masks (shape: H×W×8)
- `Landsat_test.npy`: Test data

### Training Data Format
After data preparation, training pairs are stored as:
- `Train_Pairs_Landsat/GT/`: Ground truth patches
- `Train_Pairs_Landsat/HSI/`: Degraded HSI patches
- `Train_Pairs_Landsat/OMEGA/`: Observation masks

All stored as .mat files with 256×256 patches.

## Output

### Weights
Model checkpoints saved in `Weight/` directory:
- `AGTC-Landsat_epoch_N.pth`: Intermediate checkpoints
- `AGTC-Landsat.pth`: Final trained model

### Metrics
Training and testing metrics saved in `Metrics/` directory:
- `training_metrics_YYYYMMDD_HHMMSS.csv`: Training loss over time
- `training_metrics_YYYYMMDD_HHMMSS.json`: Epoch-wise metrics
- `test_metrics_YYYYMMDD_HHMMSS.json`: Test results (PSNR, SSIM, timing)
- `Landsat-AGTC.npy`: Restored hyperspectral image

## Model Architecture
The AGTC model uses:
- **RPCA-Net**: Robust PCA-based deep unrolling network
- **Iterations**: 10 unrolled optimization blocks
- **Input**: 8-band Landsat hyperspectral patches
- **Patch Size**: 256×256
- **Loss**: L1 loss
- **Optimizer**: Adam (lr=3e-5)

## Advanced Usage

### Resume Training
```bash
python train.py --data_path ./Data-Preparation/Train_Pairs_Landsat \
                --resume ./Weight/AGTC-Landsat_epoch_50.pth \
                --N_iter 10 --input_dim 8
```

### Custom Learning Rate
```bash
python train.py --data_path ./Data-Preparation/Train_Pairs_Landsat \
                --N_iter 10 --input_dim 8 --set_lr 1e-4
```

### Test with Ground Truth
```bash
python test.py --ckpt_path ./Weight/AGTC-Landsat.pth \
               --ground_truth ../New\ folder/landsat/Landsat_ground_truth.npy
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size (currently set to 1)
- Use gradient accumulation
- Process on CPU (slower): Remove `.cuda()` calls

### Data Loading Errors
- Verify data paths in "New folder/landsat/"
- Check .npy file formats and shapes
- Ensure proper normalization (data should be in [0, 1] range)

### Training Not Converging
- Check learning rate (default: 3e-5)
- Verify data quality and normalization
- Increase number of training epochs

## Performance Metrics
The model evaluates:
- **PSNR** (Peak Signal-to-Noise Ratio): Image quality metric
- **SSIM** (Structural Similarity Index): Perceptual quality metric
- **Inference Time**: Processing speed per patch

## Citation
If you use this code, please cite the original AGTC paper:
```
@article{agtc2024,
  title={Adaptive Graph Tensor Completion for Hyperspectral Image Restoration},
  journal={...},
  year={2024}
}
```

## Support
For issues or questions:
1. Check this guide first
2. Verify all file paths and formats
3. Review error messages carefully
4. Check CUDA/GPU availability for training
