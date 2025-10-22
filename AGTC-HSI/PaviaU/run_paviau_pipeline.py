"""
Complete pipeline script for PaviaU hyperspectral image restoration
Handles data preparation, training, and testing
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_cuda():
    """Check if CUDA is available"""
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("✗ CUDA is not available. Training will be slow on CPU.")
        return False

def prepare_data():
    """Run data preparation script"""
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    data_prep_script = Path(__file__).parent / "Data-Preparation" / "train_pairs_creator_PaviaU.py"
    
    if not data_prep_script.exists():
        print(f"✗ Data preparation script not found: {data_prep_script}")
        return False
    
    print(f"Running data preparation script...")
    os.chdir(Path(__file__).parent / "Data-Preparation")
    
    try:
        result = subprocess.run([sys.executable, str(data_prep_script)], check=True)
        print("✓ Data preparation completed successfully!")
        os.chdir(Path(__file__).parent)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Data preparation failed: {e}")
        os.chdir(Path(__file__).parent)
        return False

def train_model(epochs=100, batch_size=1, checkpoint_freq=2):
    """Train the model"""
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    train_script = Path(__file__).parent / "train.py"
    train_data_path = Path(__file__).parent / "Data-Preparation" / "Train_Pairs_PaviaU"
    
    if not train_script.exists():
        print(f"✗ Training script not found: {train_script}")
        return False
    
    if not train_data_path.exists():
        print(f"✗ Training data not found: {train_data_path}")
        print("  Please run data preparation first!")
        return False
    
    print(f"Training with:")
    print(f"  - Data path: {train_data_path}")
    print(f"  - Checkpoint frequency: every {checkpoint_freq} epochs")
    print(f"  - Input dimensions: 103 (PaviaU bands)")
    print(f"  - Iterations: 10")
    
    cmd = [
        sys.executable,
        str(train_script),
        "--data_path", str(train_data_path),
        "--checkpoint_freq", str(checkpoint_freq),
        "--N_iter", "10",
        "--input_dim", "103"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed: {e}")
        return False

def test_model(ground_truth_path=None):
    """Test the trained model"""
    print("\n" + "="*70)
    print("STEP 3: MODEL TESTING")
    print("="*70)
    
    test_script = Path(__file__).parent / "test.py"
    weight_path = Path(__file__).parent / "Weight" / "AGTC-Pavia.pth"
    test_data = Path(__file__).parent / "PaviaU_test.mat"
    
    if not test_script.exists():
        print(f"✗ Test script not found: {test_script}")
        return False
    
    if not weight_path.exists():
        print(f"✗ Model weights not found: {weight_path}")
        print("  Please train the model first!")
        return False
    
    if not test_data.exists():
        print(f"✗ Test data not found: {test_data}")
        return False
    
    print(f"Testing with:")
    print(f"  - Checkpoint: {weight_path}")
    print(f"  - Test data: {test_data}")
    print(f"  - Input dimensions: 103")
    
    cmd = [
        sys.executable,
        str(test_script),
        "--ckpt_path", str(weight_path),
        "--N_iter", "10",
        "--input_dim", "103",
        "--test_data", str(test_data)
    ]
    
    if ground_truth_path:
        cmd.extend(["--ground_truth", str(ground_truth_path)])
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Testing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Testing failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline for PaviaU hyperspectral image restoration"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "prepare", "train", "test"],
        default="all",
        help="Pipeline mode: all (complete pipeline), prepare (data only), train (training only), test (testing only)"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=2,
        help="Save checkpoint every N epochs (default: 2)"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Path to ground truth for metric calculation (optional)"
    )
    parser.add_argument(
        "--skip_cuda_check",
        action="store_true",
        help="Skip CUDA availability check"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PAVIAU HYPERSPECTRAL IMAGE RESTORATION PIPELINE")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    print("="*70)
    
    # Check CUDA
    if not args.skip_cuda_check:
        check_cuda()
    
    # Run pipeline based on mode
    success = True
    
    if args.mode in ["all", "prepare"]:
        success = prepare_data()
        if not success and args.mode == "all":
            print("\n✗ Pipeline failed at data preparation stage!")
            return
    
    if args.mode in ["all", "train"] and success:
        success = train_model(checkpoint_freq=args.checkpoint_freq)
        if not success and args.mode == "all":
            print("\n✗ Pipeline failed at training stage!")
            return
    
    if args.mode in ["all", "test"] and success:
        success = test_model(ground_truth_path=args.ground_truth)
    
    # Final summary
    print("\n" + "="*70)
    if success:
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nResults saved to:")
        print(f"  - Weights: {Path(__file__).parent / 'Weight'}")
        print(f"  - Metrics: {Path(__file__).parent / 'Metrics'}")
    else:
        print("✗ PIPELINE FAILED!")
        print("="*70)
    print()

if __name__ == "__main__":
    main()
