"""
Comprehensive Setup and Run Script for AGTC Project
This script handles:
1. Environment setup
2. Data preparation
3. Model training
4. Testing
"""
import os
import sys
import subprocess
import argparse

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major != 3 or version.minor < 9:
        print("WARNING: This project was designed for Python 3.9")
        print("You may encounter compatibility issues with PyTorch 1.11")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

def install_dependencies():
    """Install required packages"""
    print_header("Installing Dependencies")
    print("Installing PyTorch 1.11 with CUDA 11.3 and other dependencies...")
    print("\nNote: This may take several minutes...\n")
    
    # Install PyTorch with CUDA 11.3
    torch_cmd = [
        sys.executable, "-m", "pip", "install",
        "torch==1.11.0+cu113",
        "torchvision==0.12.0+cu113", 
        "torchaudio==0.11.0+cu113",
        "--extra-index-url", "https://download.pytorch.org/whl/cu113"
    ]
    
    # Install other dependencies
    other_cmd = [
        sys.executable, "-m", "pip", "install",
        "numpy==1.21.5", "h5py==3.6.0", "pillow==9.0.1",
        "opencv-python==4.5.5.64", "tqdm==4.64.0", 
        "torchinfo==1.7.0", "scipy"
    ]
    
    try:
        print("Installing PyTorch...")
        subprocess.check_call(torch_cmd)
        print("\nInstalling other dependencies...")
        subprocess.check_call(other_cmd)
        print("\n✓ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        return False
    return True

def prepare_data(dataset='PaviaU'):
    """Prepare training data"""
    print_header(f"Preparing {dataset} Training Data")
    
    if dataset == 'PaviaU':
        script_path = os.path.join('AGTC-HSI', 'PaviaU', 'Data-Preparation')
        script_name = 'train_pairs_creator_PaviaU.py'
    else:  # Landsat
        script_path = os.path.join('AGTC-HSI', 'Landsat', 'Data-Preparation')
        script_name = 'train_pairs_creator_Landsat.py'
    
    original_dir = os.getcwd()
    try:
        os.chdir(script_path)
        print(f"Running data preparation script: {script_name}")
        subprocess.check_call([sys.executable, script_name])
        print(f"\n✓ {dataset} training data prepared successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error preparing data: {e}")
        return False
    finally:
        os.chdir(original_dir)

def train_model(dataset='PaviaU', use_pretrained=False):
    """Train the model"""
    print_header(f"Training Model on {dataset} Dataset")
    
    if use_pretrained:
        print("Using pretrained weights - skipping training")
        return True
    
    if dataset == 'PaviaU':
        work_dir = os.path.join('AGTC-HSI', 'PaviaU')
        data_path = './Data-Preparation/Train_Pairs_PaviaU'
    else:  # Landsat
        work_dir = os.path.join('AGTC-HSI', 'Landsat')
        data_path = './Data-Preparation/Train_Pairs_Landsat'
    
    original_dir = os.getcwd()
    try:
        os.chdir(work_dir)
        
        print("""
Training will be done in stages with learning rate adjustments:
- Epochs 1-20: lr=3e-5 (default)
- Epochs 21-40: lr=1e-6
- Epochs 41-60: lr=1e-7
- Epochs 61-80: lr=1e-8
- Epochs 81-100: lr=1e-9

Training will run for 20 epochs at a time. You'll need to confirm
continuation between stages.
        """)
        
        # Stage 1: Epochs 1-20
        print("\n--- Stage 1: Training Epochs 1-20 (lr=3e-5) ---")
        cmd1 = [sys.executable, "train.py", f"--data_path={data_path}"]
        subprocess.check_call(cmd1)
        
        # Ask to continue
        response = input("\nStage 1 complete. Continue to Stage 2? (y/n): ")
        if response.lower() != 'y':
            return False
        
        # Stage 2: Epochs 21-40
        print("\n--- Stage 2: Training Epochs 21-40 (lr=1e-6) ---")
        cmd2 = [sys.executable, "train.py", f"--data_path={data_path}",
                "--resume=./checkpoints/epoch_20.pth", "--set_lr=1e-6"]
        subprocess.check_call(cmd2)
        
        response = input("\nStage 2 complete. Continue to Stage 3? (y/n): ")
        if response.lower() != 'y':
            return False
        
        # Stage 3: Epochs 41-60
        print("\n--- Stage 3: Training Epochs 41-60 (lr=1e-7) ---")
        cmd3 = [sys.executable, "train.py", f"--data_path={data_path}",
                "--resume=./checkpoints/epoch_40.pth", "--set_lr=1e-7"]
        subprocess.check_call(cmd3)
        
        response = input("\nStage 3 complete. Continue to Stage 4? (y/n): ")
        if response.lower() != 'y':
            return False
        
        # Stage 4: Epochs 61-80
        print("\n--- Stage 4: Training Epochs 61-80 (lr=1e-8) ---")
        cmd4 = [sys.executable, "train.py", f"--data_path={data_path}",
                "--resume=./checkpoints/epoch_60.pth", "--set_lr=1e-8"]
        subprocess.check_call(cmd4)
        
        response = input("\nStage 4 complete. Continue to Stage 5 (final)? (y/n): ")
        if response.lower() != 'y':
            return False
        
        # Stage 5: Epochs 81-100
        print("\n--- Stage 5: Training Epochs 81-100 (lr=1e-9) ---")
        cmd5 = [sys.executable, "train.py", f"--data_path={data_path}",
                "--resume=./checkpoints/epoch_80.pth", "--set_lr=1e-9"]
        subprocess.check_call(cmd5)
        
        print("\n✓ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        return False
    finally:
        os.chdir(original_dir)

def test_model(dataset='PaviaU'):
    """Test the trained model"""
    print_header(f"Testing Model on {dataset} Dataset")
    
    if dataset == 'PaviaU':
        work_dir = os.path.join('AGTC-HSI', 'PaviaU')
        output_file = 'Pavia-AGTC.npy'
    else:  # Landsat
        work_dir = os.path.join('AGTC-HSI', 'Landsat')
        output_file = 'Landsat-AGTC.npy'
    
    original_dir = os.getcwd()
    try:
        os.chdir(work_dir)
        print("Running test script...")
        subprocess.check_call([sys.executable, "test.py"])
        print(f"\n✓ Testing completed! Output saved to: {output_file}")
        return True
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        return False
    finally:
        os.chdir(original_dir)

def main():
    parser = argparse.ArgumentParser(description='AGTC Project Setup and Run Script')
    parser.add_argument('--dataset', choices=['PaviaU', 'Landsat'], default='PaviaU',
                       help='Dataset to use (default: PaviaU)')
    parser.add_argument('--skip-install', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--skip-data-prep', action='store_true',
                       help='Skip data preparation')
    parser.add_argument('--use-pretrained', action='store_true',
                       help='Use pretrained weights (skip training)')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing (assumes pretrained weights)')
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    AGTC PROJECT SETUP & RUN                       ║
║   Attention-Guided Low-Rank Tensor Completion                     ║
║   for Hyperspectral Image Restoration                             ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not args.skip_install and not args.test_only:
        if not install_dependencies():
            print("\n✗ Setup failed during dependency installation")
            return
    
    # Prepare data
    if not args.skip_data_prep and not args.test_only:
        if not prepare_data(args.dataset):
            print("\n✗ Setup failed during data preparation")
            return
    
    # Train model
    if not args.test_only:
        if not train_model(args.dataset, args.use_pretrained):
            print("\n✗ Training failed or was cancelled")
            return
    
    # Test model
    if not test_model(args.dataset):
        print("\n✗ Testing failed")
        return
    
    print_header("COMPLETE!")
    print(f"""
✓ All steps completed successfully for {args.dataset} dataset!

The restored hyperspectral image has been generated.
Check the AGTC-HSI/{args.dataset}/ directory for output.
    """)

if __name__ == '__main__':
    main()
