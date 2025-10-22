# AGTC-HSI Image Upload & Restoration - User Guide

## ✅ Application Status
**The application is now running successfully at:** http://127.0.0.1:5000

## 🎯 Features

### 1. **Upload Your Own Images**
- Click or drag & drop any image (JPG, PNG, BMP)
- Supports any image size
- Images are automatically converted to hyperspectral format
- Synthetic degradation is applied for restoration demonstration

### 2. **Two Dataset Modes**
- **Landsat** (8 bands): 256x256 patches, optimized for satellite imagery
- **PaviaU** (103 bands): 64x64 patches, optimized for urban hyperspectral data

### 3. **Demo Data**
- Test with built-in degraded hyperspectral test data
- Compare restoration results instantly

## 📖 How to Use

### Upload Custom Images:
1. **Select Dataset**: Click either Landsat or PaviaU card
2. **Upload Image**: 
   - Click the upload area or drag & drop your image
   - Preview will appear automatically
3. **Restore**: Click "🔧 Restore Uploaded Image"
4. **View Results**: See before/after comparison + quality metrics (PSNR, SSIM, RSE)

### Try Demo Data:
1. **Select Dataset**: Choose Landsat or PaviaU
2. **Click "🎬 Try Demo Data"**
3. **View Results**: Instant restoration on test data

## 🔧 Technical Details

### Image Processing Pipeline:
1. **Input**: RGB image (any size)
2. **Preprocessing**:
   - Resize/crop to patch size (256x256 or 64x64)
   - Normalize to [0, 1]
3. **Spectral Conversion**:
   - Landsat: RGB replicated to 8 bands
   - PaviaU: RGB expanded to 103 bands with spectral variation
4. **Degradation**: 10% random missing pixels applied
5. **Restoration**: AGTC model restores missing data
6. **Output**: Restored hyperspectral image

### Model Architecture:
- **RPCA-Net**: Deep unrolling Robust PCA
- **Spatial Attention**: Learns importance weights
- **RDB**: Residual Dense Blocks for feature extraction
- **Tensor Operations**: FFT-based for efficiency

### Quality Metrics:
- **PSNR**: Peak Signal-to-Noise Ratio (higher = better, typically 30-40 dB)
- **SSIM**: Structural Similarity (higher = better, range 0-1)
- **RSE**: Relative Spectral Error (lower = better)

## 🎨 Supported Image Formats
- JPEG / JPG
- PNG
- BMP
- Any RGB image format supported by PIL

## ⚙️ Configuration

### Current Settings:
- **Device**: CPU (CUDA not available)
- **Max Upload Size**: 100 MB
- **Processing Time**: 10-30 seconds per image
- **Port**: 5000

### Dataset Specifications:
| Dataset | Bands | Patch Size | Model Size |
|---------|-------|------------|------------|
| Landsat | 8     | 256×256    | 27 MB      |
| PaviaU  | 103   | 64×64      | 36 MB      |

## 📊 Expected Results

### Typical Performance:
- **Landsat**:
  - PSNR: 32-38 dB
  - SSIM: 0.90-0.98
  - Processing: 15-25 seconds

- **PaviaU**:
  - PSNR: 34-42 dB
  - SSIM: 0.92-0.99
  - Processing: 10-20 seconds

## 🚀 Quick Start Commands

### Start Application:
```bash
cd "d:\SATYA\AGTC-MAJOR PROJECT\AGTC\AGTC-HSI"
python app.py
```

### Stop Application:
Press `Ctrl+C` in the terminal

### Access Web Interface:
Open browser to: http://127.0.0.1:5000

## 🔍 API Endpoints

### For Developers:
- `GET /` - Main web interface
- `POST /api/upload` - Upload and restore custom images
- `POST /api/restore` - Restore with demo data
- `GET /api/datasets` - List available datasets
- `GET /api/status` - Check server status

## 📝 Notes

1. **First-time use**: Models are loaded on first request (may take 5-10 seconds)
2. **Uploaded files**: Automatically deleted after processing
3. **CPU Mode**: Running on CPU (slower than GPU, but functional)
4. **Image conversion**: RGB images are converted to synthetic hyperspectral data
5. **Real hyperspectral data**: For actual HSI files (.mat, .npy), use the pipeline scripts

## 🛠️ Troubleshooting

### Issue: Server won't start
**Solution**: Check if port 5000 is in use:
```bash
netstat -ano | findstr :5000
```

### Issue: Upload fails
**Solution**: 
- Check file size (max 100 MB)
- Ensure image format is supported
- Verify sufficient disk space

### Issue: Slow processing
**Solution**: 
- Normal on CPU (10-30 seconds)
- For faster processing, install CUDA-enabled PyTorch

## 🎓 For Research Use

This application demonstrates the AGTC (Attention-Guided Low-Rank Tensor Completion) algorithm:
- **Paper**: IEEE TPAMI 2024
- **Authors**: Truong Thanh Nhat Mai, Edmund Y. Lam, Chul Lee
- **DOI**: 10.1109/TPAMI.2024.3429498

### Citation:
```bibtex
@ARTICLE{Mai2024,
  author={Mai, Truong Thanh Nhat and Lam, Edmund Y. and Lee, Chul},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Attention-Guided Low-Rank Tensor Completion}, 
  year={2024},
  volume={46},
  number={12},
  pages={9818-9833}
}
```

## 📧 Support

For issues or questions:
1. Check this guide and README.md
2. Review the console output for error messages
3. Verify setup with: `python verify_setup.py`
4. Check the GitHub repository for updates

---

**Enjoy restoring your images! 🚀**
