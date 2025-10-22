"""
Python equivalent of train_pairs_creator_Landsat.m
Creates training pairs for Landsat dataset
Updated to use .npy files from New folder
"""
import numpy as np
import scipy.io as sio
import os
from tqdm import tqdm

def create_training_pairs():
    """
    Main function to create training pairs for Landsat
    """
    print("Loading Landsat training data...")
    
    # Load data from New folder (using .npy files)
    new_folder_path = os.path.join('..', 'New folder', 'landsat')
    clean_data = np.load(os.path.join(new_folder_path, 'Landsat7_training_clean.npy'))
    mask_data = np.load(os.path.join(new_folder_path, 'Landsat7_training_mask.npy'))
    
    # Normalize the data
    gt = clean_data.astype(np.float64)
    if gt.max() > 1.0:
        gt = gt / gt.max()  # Normalize to [0, 1] range
    mask = mask_data.astype(np.float64)
    hsi = gt * mask
    
    print(f"Ground truth shape: {gt.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Data range - GT: [{gt.min():.4f}, {gt.max():.4f}], Mask: [{mask.min():.4f}, {mask.max():.4f}]")
    
    # Create directories
    os.makedirs('./Train_Pairs_Landsat/GT/', exist_ok=True)
    os.makedirs('./Train_Pairs_Landsat/HSI/', exist_ok=True)
    os.makedirs('./Train_Pairs_Landsat/OMEGA/', exist_ok=True)
    
    mat_idx = -1
    
    print("Creating training pairs...")
    # Calculate valid range for patch extraction (256x256 patches)
    max_x = gt.shape[0] - 256
    max_y = gt.shape[1] - 256
    print(f"Valid patch extraction range: X=[0, {max_x}], Y=[0, {max_y}]")
    
    for idx in tqdm(range(4500)):
        # Random patch location (Python uses 0-indexing, so adjust ranges)
        pix_x = np.random.randint(0, max_x)
        pix_y = np.random.randint(0, max_y)
        
        mat_idx += 1
        mat_name = f'{mat_idx:06d}'
        
        # Extract patches (256x256)
        gt_aug = gt[pix_x:pix_x+256, pix_y:pix_y+256, :]
        Nmsi = hsi[pix_x:pix_x+256, pix_y:pix_y+256, :]
        Omega3_3D = mask[pix_x:pix_x+256, pix_y:pix_y+256, :]
        
        # Save patches
        sio.savemat(f'./Train_Pairs_Landsat/GT/{mat_name}.mat', 
                   {'gt_aug': gt_aug}, do_compression=True)
        sio.savemat(f'./Train_Pairs_Landsat/HSI/{mat_name}.mat', 
                   {'Nmsi': Nmsi}, do_compression=True)
        sio.savemat(f'./Train_Pairs_Landsat/OMEGA/{mat_name}.mat', 
                   {'Omega3_3D': Omega3_3D}, do_compression=True)
    
    print(f"Created {mat_idx + 1} training pairs successfully!")

if __name__ == '__main__':
    create_training_pairs()
