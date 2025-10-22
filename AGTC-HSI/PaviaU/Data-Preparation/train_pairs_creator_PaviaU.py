"""
Python equivalent of train_pairs_creator_PaviaU.m
Creates training pairs for PaviaU dataset with data augmentation
"""
import numpy as np
import scipy.io as sio
import os
from tqdm import tqdm

def patch_augmentation(input_patch, ind):
    """
    Data augmentation function - equivalent to PatchAugmentation.m
    """
    if ind == 1:
        return input_patch
    elif ind == 2:
        return np.rot90(input_patch, k=1, axes=(0, 1))
    elif ind == 3:
        return np.rot90(input_patch, k=2, axes=(0, 1))
    elif ind == 4:
        return np.fliplr(input_patch)
    elif ind == 5:
        return np.flipud(input_patch)
    elif ind == 6:
        return np.rot90(np.fliplr(input_patch), k=1, axes=(0, 1))
    elif ind == 7:
        return np.rot90(np.fliplr(input_patch), k=1, axes=(0, 1))
    elif ind == 8:
        return np.rot90(np.flipud(input_patch), k=2, axes=(0, 1))
    elif ind == 9:
        return np.rot90(np.flipud(input_patch), k=2, axes=(0, 1))
    else:
        return input_patch

def make_stripes(original_image, d, r):
    """
    Add stripe noise to image - equivalent to make_stripes.m
    """
    num_rows, num_cols, num_bands = original_image.shape
    
    # Reshape to 2D for easier processing
    image_2d = original_image.reshape(num_rows, num_cols * num_bands)
    
    # Determine number of stripes
    num_stripes = round(d * num_cols * num_bands)
    
    # Random stripe locations
    stripe_locations = np.random.permutation(num_cols * num_bands)[:num_stripes]
    
    # Create stripes
    stripes = r * (2 * np.random.rand(num_rows, num_stripes) - 1)
    
    # Apply stripes
    striped_image_2d = image_2d.copy()
    striped_image_2d[:, stripe_locations] = image_2d[:, stripe_locations] + stripes
    
    # Clip values
    striped_image_2d = np.clip(striped_image_2d, 1e-3, 1 - 0.001)
    
    # Reshape back to 3D
    striped_image = striped_image_2d.reshape(num_rows, num_cols, num_bands)
    
    return striped_image, stripe_locations

def create_training_pairs():
    """
    Main function to create training pairs
    """
    print("Loading PaviaU data from New folder...")
    new_folder_path = os.path.join('..', 'New folder', 'paviau')
    paviau = np.load(os.path.join(new_folder_path, 'PaviaU.npy')).astype(np.float32)
    
    print(f"PaviaU data shape: {paviau.shape}")
    
    # Normalize
    paviau = paviau / np.max(paviau)
    print(f"Data range after normalization: [{paviau.min():.4f}, {paviau.max():.4f}]")
    
    # Parameters
    rate = 0.5
    mean_val = np.mean(paviau[0:min(256, paviau.shape[0]), 0:min(256, paviau.shape[1]), :])
    M, N, p = 64, 64, paviau.shape[2]
    print(f"Patch size: {M}x{N}, Number of bands: {p}")
    
    # Create directories
    os.makedirs('./Train_Pairs_PaviaU/GT/', exist_ok=True)
    os.makedirs('./Train_Pairs_PaviaU/HSI/', exist_ok=True)
    os.makedirs('./Train_Pairs_PaviaU/OMEGA/', exist_ok=True)
    
    mat_idx = -1
    
    print("Creating training pairs...")
    # Calculate valid range for patch extraction (64x64 patches)
    max_x = max(64, paviau.shape[0] - 64)
    max_y = max(64, paviau.shape[1] - 64)
    print(f"Valid patch extraction range: X=[0, {max_x}], Y=[0, {max_y}]")
    
    for idx in tqdm(range(500)):
        # Random patch location (Python uses 0-indexing)
        pix_x = np.random.randint(0, max_x)
        pix_y = np.random.randint(0, max_y)
        gt_msi = paviau[pix_x:pix_x+64, pix_y:pix_y+64, :]
        
        # Data augmentation
        for aug in range(1, 10):
            mat_idx += 1
            mat_name = f'{mat_idx:06d}'
            
            # Augment patch
            gt_aug = patch_augmentation(gt_msi, aug)
            
            # Save ground truth
            sio.savemat(f'./Train_Pairs_PaviaU/GT/{mat_name}.mat', 
                       {'gt_aug': gt_aug}, do_compression=True)
            
            # Add stripes
            Nmsi, loc = make_stripes(gt_aug, rate, mean_val)
            
            # Create omega mask
            Omega3 = np.ones((M, N * p), dtype=np.float32)
            Omega3[:, loc] = 0
            Omega3_3D = Omega3.reshape(M, N, p)
            
            # Remove some columns completely
            completely_gone = np.random.randint(0, 64, size=2)
            Omega3_3D[:, completely_gone, :] = 0
            
            # Save degraded image and mask
            sio.savemat(f'./Train_Pairs_PaviaU/HSI/{mat_name}.mat', 
                       {'Nmsi': Nmsi}, do_compression=True)
            sio.savemat(f'./Train_Pairs_PaviaU/OMEGA/{mat_name}.mat', 
                       {'Omega3_3D': Omega3_3D}, do_compression=True)
    
    print(f"Created {mat_idx + 1} training pairs successfully!")

if __name__ == '__main__':
    create_training_pairs()
