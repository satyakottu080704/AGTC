# AGTC-HSI Quick Start Guide

## 🚀 5-Minute Setup

### 1. Install Dependencies (One Time)
```bash
cd "d:\SATYA\AGTC-MAJOR PROJECT\AGTC\AGTC-HSI"
pip install -r requirements.txt
```

### 2. Run Landsat Pipeline
```bash
cd Landsat
python run_landsat_pipeline.py --mode all
```
**Expected Time**: ~2-4 hours (depending on GPU)

### 3. Run PaviaU Pipeline
```bash
cd PaviaU
python run_paviau_pipeline.py --mode all
```
**Expected Time**: ~2-4 hours (depending on GPU)

## 📊 What Gets Created

After running the pipeline, you'll find:

```
Landsat/
├── Weight/
│   └── AGTC-Landsat.pth          ← Trained model
└── Metrics/
    ├── training_metrics_*.csv     ← Training loss log
    ├── training_metrics_*.json    ← Epoch summaries
    ├── test_metrics_*.json        ← Test results (PSNR/SSIM)
    └── Landsat-AGTC.npy          ← Restored image

PaviaU/
├── Weight/
│   └── AGTC-Pavia.pth            ← Trained model
└── Metrics/
    ├── training_metrics_*.csv     ← Training loss log
    ├── training_metrics_*.json    ← Epoch summaries
    ├── test_metrics_*.json        ← Test results (PSNR/SSIM)
    └── Pavia-AGTC.npy            ← Restored image
```

## 🎯 Run Individual Steps

If you want more control:

### Data Preparation Only
```bash
python run_landsat_pipeline.py --mode prepare
python run_paviau_pipeline.py --mode prepare
```

### Training Only
```bash
python run_landsat_pipeline.py --mode train
python run_paviau_pipeline.py --mode train
```

### Testing Only
```bash
python run_landsat_pipeline.py --mode test
python run_paviau_pipeline.py --mode test
```

## 🔧 Common Commands

**Save checkpoints more frequently:**
```bash
python run_landsat_pipeline.py --mode train --checkpoint_freq 5
```

**Skip CUDA check:**
```bash
python run_landsat_pipeline.py --mode all --skip_cuda_check
```

**Test with ground truth for metrics:**
```bash
cd Landsat
python test.py --ground_truth "../New folder/landsat/Landsat_ground_truth.npy"
```

## ⚡ System Requirements

- **GPU**: NVIDIA GPU with CUDA (strongly recommended)
- **RAM**: 8GB minimum
- **Disk**: 10GB free space
- **Python**: 3.7 or higher

## 🆘 Troubleshooting

**CUDA not found?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory?**
- Close other applications
- Use CPU mode (slower): Add `--skip_cuda_check`

**Data not found?**
- Check `New folder/landsat/` has .npy files
- Check `New folder/paviau/` has .npy files

## 📖 Need More Details?

- **Landsat**: See `Landsat/USAGE_GUIDE.md`
- **PaviaU**: See `PaviaU/USAGE_GUIDE.md`
- **Overview**: See `README.md`

## ✅ Verify Installation

```bash
# Test Python and PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test other dependencies
python -c "import numpy, scipy, h5py, tqdm, torchinfo, skimage; print('All dependencies installed!')"
```

## 🎓 What This Does

1. **Data Preparation**: Converts .npy files from "New folder" into training patches
2. **Training**: Trains RPCA-Net for 100 epochs with checkpointing
3. **Testing**: Evaluates model on test data and calculates metrics

**Landsat**: 8 bands, 256×256 patches, ~4,500 training samples  
**PaviaU**: 103 bands, 64×64 patches, ~4,500 training samples (with augmentation)

---

**That's it!** The model is ready to restore hyperspectral images. 🎉
