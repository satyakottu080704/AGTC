import os
import cv2
import torch
import copy
import time
import argparse
import h5py
import json
import numpy as np
from datetime import datetime
from main_net import RPCA_Net
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_pretrained(path, N_iter, input_dim):
    model = RPCA_Net(N_iter=N_iter, tensor_num_channels=input_dim)
    model = model.cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def inference(model, img, omg):
    img0 = torch.unsqueeze(img.cuda(), 0)
    omg0 = torch.unsqueeze(omg.cuda(), 0)

    with torch.no_grad():
        C = model(img0, omg0)

    hsi_patch = torch.squeeze(C).cpu().numpy()
    return hsi_patch


def create_images(ckpt_path, N_iter, input_dim, test_data_path='PaviaU_test.mat', ground_truth_path=None):
    print(f"Starting inference with checkpoint: {ckpt_path}")
    
    # Load model
    model = load_pretrained(ckpt_path, N_iter, input_dim)

    # Grid
    w_grid = [0, 64, 128, 192]
    h_grid = [0, 64, 128, 192]

    w_grid = np.array(w_grid, dtype=np.uint16)
    h_grid = np.array(h_grid, dtype=np.uint16)

    # Load test data
    print(f"Loading test data from {test_data_path}...")
    f = h5py.File(test_data_path, 'r')
    reader = f.get('Nmsi')
    data = torch.from_numpy(np.array(reader).astype('float32'))
    reader = f.get('mask')
    omega = torch.from_numpy(np.array(reader).astype('float32'))
    f.close()
    
    print(f"Data shape: {data.shape}, Omega shape: {omega.shape}")

    HSI = np.float32(np.zeros((256, 256, 103)))
    
    # Try to load ground truth if available
    ground_truth = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        try:
            if ground_truth_path.endswith('.npy'):
                ground_truth = np.load(ground_truth_path)
            elif ground_truth_path.endswith('.mat'):
                gt_f = h5py.File(ground_truth_path, 'r')
                ground_truth = np.array(gt_f.get('ground_truth')).astype('float32')
                gt_f.close()
            print(f"Ground truth loaded: {ground_truth.shape}")
        except Exception as e:
            print(f"Could not load ground truth: {e}")

    # Patch reconstruction
    print("Starting patch-based reconstruction...")
    i = 0
    j = 0
    total_time = 0
    patch_times = []
    
    while i < len(h_grid):
        while j < len(w_grid):
            h = h_grid[i]
            w = w_grid[j]

            data_patch = copy.deepcopy(data[:, w:w + 64, h:h + 64])
            omega_patch = copy.deepcopy(omega[:, w:w + 64, h:h + 64])
            
            # Patch inference and reshape
            patch_start = time.time()
            C = inference(model, data_patch, omega_patch)
            patch_time = time.time() - patch_start
            patch_times.append(patch_time)
            total_time += patch_time

            hsi_patch = C.transpose(2, 1, 0)

            # Stitching
            HSI[h:h + 64, w:w + 64, :] = copy.deepcopy(hsi_patch)

            j = j + 1
        i = i + 1
        j = 0
    
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Average patch time: {np.mean(patch_times):.4f}s")
    
    # Save result
    output_path = './Metrics/Pavia-AGTC.npy'
    os.makedirs('./Metrics', exist_ok=True)
    np.save(output_path, HSI)
    print(f"Result saved to {output_path}")
    
    # Calculate metrics if ground truth is available
    metrics = {
        'total_time': total_time,
        'avg_patch_time': float(np.mean(patch_times)),
        'num_patches': len(patch_times),
        'timestamp': datetime.now().isoformat()
    }
    
    if ground_truth is not None:
        try:
            # Ensure same shape
            if ground_truth.shape == HSI.shape:
                psnr_val = psnr(ground_truth, HSI, data_range=1.0)
                ssim_val = ssim(ground_truth, HSI, multichannel=True, channel_axis=2, data_range=1.0)
                metrics['psnr'] = float(psnr_val)
                metrics['ssim'] = float(ssim_val)
                print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
            else:
                print(f"Shape mismatch: GT {ground_truth.shape} vs Result {HSI.shape}")
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    
    # Save metrics
    metrics_file = f'./Metrics/test_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")
    
    return HSI, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default='./Weight/AGTC-Pavia.pth', help='Path to checkpoint')
    parser.add_argument('--N_iter', type=int, default=10, help='Number of iterations')
    parser.add_argument('--input_dim', type=int, default=103, help='Input dimensions')
    parser.add_argument('--test_data', default='PaviaU_test.mat', help='Test data path')
    parser.add_argument('--ground_truth', default=None, help='Ground truth path (optional)')
    args = parser.parse_args()
    
    create_images(
        ckpt_path=args.ckpt_path,
        N_iter=args.N_iter,
        input_dim=args.input_dim,
        test_data_path=args.test_data,
        ground_truth_path=args.ground_truth
    )

    print('Done!')
