# AGTC-HSI: Hyperspectral Image Restoration

## Overview
This project implements Adaptive Graph Tensor Completion (AGTC) for hyperspectral image restoration. The model handles missing data, stripe noise, and other degradations in hyperspectral images using a deep unrolling RPCA-based approach.

## Supported Datasets
- **Landsat**: 8-band hyperspectral imagery
- **PaviaU**: 103-band hyperspectral imagery

## Project Structure
```
AGTC-HSI/
├── Landsat/                    # Landsat dataset pipeline
│   ├── Data-Preparation/       # Data preprocessing
│   ├── Weight/                 # Model weights
│   ├── Metrics/                # Training & testing metrics
│   ├── run_landsat_pipeline.py # Main executable
│   ├── train.py               # Training script
│   ├── test.py                # Testing script
│   ├── main_net.py            # Model architecture
│   └── USAGE_GUIDE.md         # Detailed guide
│
├── PaviaU/                     # PaviaU dataset pipeline
│   ├── Data-Preparation/       # Data preprocessing
│   ├── Weight/                 # Model weights
│   ├── Metrics/                # Training & testing metrics
│   ├── run_paviau_pipeline.py  # Main executable
│   ├── train.py               # Training script
│   ├── test.py                # Testing script
│   ├── main_net.py            # Model architecture
│   └── USAGE_GUIDE.md         # Detailed guide
│
├── New folder/                 # Source data
│   ├── landsat/               # Landsat raw data (.npy files)
│   └── paviau/                # PaviaU raw data (.npy files)
│
└── requirements.txt           # Python dependencies
```

## Features
✅ **Complete Pipeline**: Data preparation → Training → Testing  
✅ **Automated Metrics**: PSNR, SSIM, inference time tracking  
✅ **Checkpoint Management**: Save/resume training from any epoch  
✅ **CSV & JSON Logging**: Detailed training and testing metrics  
✅ **Flexible Configuration**: Command-line arguments for all parameters  
✅ **GPU Acceleration**: CUDA support for fast training  
✅ **Data Augmentation**: Built-in augmentation for PaviaU dataset  

## Quick Start

### 1. Installation
```bash
# Clone or navigate to the project directory
cd "d:\SATYA\AGTC-MAJOR PROJECT\AGTC\AGTC-HSI"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Landsat Pipeline
```bash
cd Landsat
python run_landsat_pipeline.py --mode all
```

### 3. Run PaviaU Pipeline
```bash
cd PaviaU
python run_paviau_pipeline.py --mode all
```

## Detailed Usage

### Landsat Dataset
```bash
cd Landsat

# Complete pipeline
python run_landsat_pipeline.py --mode all

# Individual steps
python run_landsat_pipeline.py --mode prepare  # Data preparation only
python run_landsat_pipeline.py --mode train    # Training only
python run_landsat_pipeline.py --mode test     # Testing only

# Custom checkpoint frequency
python run_landsat_pipeline.py --mode train --checkpoint_freq 5
```

### PaviaU Dataset
```bash
cd PaviaU

# Complete pipeline
python run_paviau_pipeline.py --mode all

# Individual steps
python run_paviau_pipeline.py --mode prepare  # Data preparation only
python run_paviau_pipeline.py --mode train    # Training only
python run_paviau_pipeline.py --mode test     # Testing only

# Custom checkpoint frequency
python run_paviau_pipeline.py --mode train --checkpoint_freq 5
```

## Model Architecture

### RPCA-Net (Robust PCA Network)
The model uses a deep unrolling approach based on Robust Principal Component Analysis:

**Key Components:**
- **Spatial Attention Module**: Learns spatial importance weights
- **RDB (Residual Dense Blocks)**: Feature extraction with dense connections
- **RPCA Block**: Unrolled optimization for low-rank + sparse decomposition
- **Tensor Operations**: FFT-based tensor product for efficient computation

**Training Details:**
- **Loss Function**: L1 Loss
- **Optimizer**: Adam (lr=3e-5)
- **Epochs**: 100
- **Batch Size**: 1
- **Gradient Clipping**: Max norm of 2

**Architecture Specifics:**
| Dataset | Input Bands | Patch Size | Iterations |
|---------|------------|------------|------------|
| Landsat | 8          | 256×256    | 10         |
| PaviaU  | 103        | 64×64      | 10         |

## Data Preparation

### Landsat Data
- **Source**: `New folder/landsat/`
- **Files**: 
  - `Landsat7_training_clean.npy`: Clean HSI
  - `Landsat7_training_mask.npy`: Observation mask
- **Output**: 4,500 training patches (256×256)

### PaviaU Data
- **Source**: `New folder/paviau/`
- **Files**: 
  - `PaviaU.npy`: Hyperspectral image
- **Output**: 4,500 training patches (64×64) with 9× augmentation
- **Degradation**: Stripe noise + random column removal

## Training Output

### Weights
Saved in `<dataset>/Weight/` directory:
- Intermediate checkpoints: `AGTC-<Dataset>_epoch_N.pth`
- Final model: `AGTC-<Dataset>.pth`

### Metrics
Saved in `<dataset>/Metrics/` directory:
- `training_metrics_*.csv`: Iteration-level loss
- `training_metrics_*.json`: Epoch-level summary
- `test_metrics_*.json`: Test results with PSNR/SSIM

## Performance Metrics

The model tracks:
- **PSNR (Peak Signal-to-Noise Ratio)**: Quantitative image quality
- **SSIM (Structural Similarity Index)**: Perceptual similarity
- **Inference Time**: Per-patch processing time
- **Training Loss**: L1 loss over epochs

## Requirements

### System Requirements
- Python 3.7+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space for checkpoints

### Software Dependencies
See `requirements.txt` for complete list:
- PyTorch ≥ 1.10.0
- NumPy, SciPy
- h5py, scikit-image
- tqdm, torchinfo

## Advanced Configuration

### Resume Training
```bash
python train.py --data_path <path> --resume <checkpoint.pth>
```

### Custom Learning Rate
```bash
python train.py --data_path <path> --set_lr 1e-4
```

### Test with Ground Truth
```bash
python test.py --ckpt_path <model.pth> --ground_truth <gt.npy>
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce batch size
- Use smaller patches
- Enable gradient checkpointing

**2. Data Loading Errors**
- Verify file paths in "New folder"
- Check .npy file formats
- Ensure data is normalized [0, 1]

**3. Import Errors**
```bash
pip install -r requirements.txt --upgrade
```

**4. Model Not Converging**
- Check learning rate
- Verify data quality
- Increase training epochs

## File Formats

### Input Data (.npy)
```python
# Landsat
clean: np.array, shape (H, W, 8), dtype float32
mask:  np.array, shape (H, W, 8), dtype float32

# PaviaU
data:  np.array, shape (H, W, 103), dtype float32
```

### Model Checkpoint (.pth)
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'loss': float  # optional
}
```

## Visualization

To visualize results:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load result
result = np.load('Metrics/Landsat-AGTC.npy')

# Plot RGB bands (if available)
plt.imshow(result[:, :, :3])
plt.title('Restored HSI (RGB)')
plt.show()
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{agtc2024,
  title={Adaptive Graph Tensor Completion for Hyperspectral Image Restoration},
  author={...},
  journal={...},
  year={2024}
}
```

## License
See LICENSE file in project root.

## Contact & Support

For questions and issues:
1. Check the USAGE_GUIDE.md in respective dataset folders
2. Review error messages and logs
3. Verify CUDA/GPU setup
4. Ensure data integrity

## Acknowledgments

This implementation is based on the AGTC (Adaptive Graph Tensor Completion) method for hyperspectral image restoration with enhancements for:
- Automated pipeline execution
- Comprehensive metrics logging
- Flexible configuration
- User-friendly interfaces
