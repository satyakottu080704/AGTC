"""
Setup Verification Script for AGTC-HSI
Checks all dependencies, file paths, and configurations
"""
import sys
import os
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_python_version():
    """Check Python version"""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 7:
        print("✅ Python version is compatible (3.7+)")
        return True
    else:
        print("❌ Python 3.7 or higher is required")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print_section("Dependencies Check")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'h5py': 'h5py',
        'tqdm': 'tqdm',
        'torchinfo': 'torchinfo',
        'skimage': 'scikit-image',
        'PIL': 'Pillow',
        'cv2': 'opencv-python'
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            if package == 'skimage':
                __import__('skimage')
            elif package == 'PIL':
                __import__('PIL')
            elif package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            print(f"✅ {name:20s} - Installed")
        except ImportError:
            print(f"❌ {name:20s} - NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print("\n⚠️  Install missing packages with:")
        print("   pip install -r requirements.txt")
    
    return all_installed

def check_cuda():
    """Check CUDA availability"""
    print_section("CUDA Check")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   PyTorch Version: {torch.__version__}")
            return True
        else:
            print("⚠️  CUDA is NOT available")
            print("   Training will run on CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch is not installed")
        return False

def check_data_files():
    """Check if data files exist"""
    print_section("Data Files Check")
    
    base_path = Path(__file__).parent
    
    # Landsat files
    landsat_files = [
        "New folder/landsat/Landsat7_training_clean.npy",
        "New folder/landsat/Landsat7_training_mask.npy",
        "Landsat/Landsat_test.mat"
    ]
    
    # PaviaU files
    paviau_files = [
        "New folder/paviau/PaviaU.npy",
        "PaviaU/PaviaU_test.mat"
    ]
    
    all_files_exist = True
    
    print("\nLandsat Dataset:")
    for file in landsat_files:
        file_path = base_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {file:50s} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {file:50s} NOT FOUND")
            all_files_exist = False
    
    print("\nPaviaU Dataset:")
    for file in paviau_files:
        file_path = base_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {file:50s} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {file:50s} NOT FOUND")
            all_files_exist = False
    
    return all_files_exist

def check_directory_structure():
    """Check if required directories exist"""
    print_section("Directory Structure")
    
    base_path = Path(__file__).parent
    
    required_dirs = [
        "Landsat",
        "Landsat/Data-Preparation",
        "PaviaU",
        "PaviaU/Data-Preparation",
        "New folder",
        "New folder/landsat",
        "New folder/paviau"
    ]
    
    all_dirs_exist = True
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - NOT FOUND")
            all_dirs_exist = False
    
    return all_dirs_exist

def check_scripts():
    """Check if all required scripts exist"""
    print_section("Scripts Check")
    
    base_path = Path(__file__).parent
    
    required_scripts = [
        "Landsat/run_landsat_pipeline.py",
        "Landsat/train.py",
        "Landsat/test.py",
        "Landsat/main_net.py",
        "Landsat/util.py",
        "Landsat/Data-Preparation/train_pairs_creator_Landsat.py",
        "PaviaU/run_paviau_pipeline.py",
        "PaviaU/train.py",
        "PaviaU/test.py",
        "PaviaU/main_net.py",
        "PaviaU/util.py",
        "PaviaU/Data-Preparation/train_pairs_creator_PaviaU.py"
    ]
    
    all_scripts_exist = True
    for script in required_scripts:
        script_path = base_path / script
        if script_path.exists():
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} - NOT FOUND")
            all_scripts_exist = False
    
    return all_scripts_exist

def print_next_steps():
    """Print next steps based on verification results"""
    print_section("Next Steps")
    
    print("""
To run the complete pipeline:

For Landsat:
  cd Landsat
  python run_landsat_pipeline.py --mode all

For PaviaU:
  cd PaviaU
  python run_paviau_pipeline.py --mode all

For more detailed instructions, see:
  - QUICKSTART.md (5-minute guide)
  - README.md (comprehensive overview)
  - Landsat/USAGE_GUIDE.md (Landsat specific)
  - PaviaU/USAGE_GUIDE.md (PaviaU specific)
""")

def main():
    """Main verification function"""
    print("\n" + "="*70)
    print("  AGTC-HSI Setup Verification")
    print("="*70)
    
    checks = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "CUDA Support": check_cuda(),
        "Data Files": check_data_files(),
        "Directory Structure": check_directory_structure(),
        "Scripts": check_scripts()
    }
    
    # Summary
    print_section("Verification Summary")
    
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check_name:20s}: {status}")
    
    print()
    
    if all(checks.values()):
        print("🎉 All checks passed! Your setup is ready.")
        print_next_steps()
    else:
        print("⚠️  Some checks failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check data file paths in 'New folder'")
        print("  - Verify all scripts were properly updated")
    
    print("="*70)
    print()

if __name__ == "__main__":
    main()
