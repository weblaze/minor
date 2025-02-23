from pathlib import Path
from normalize_features import normalize_features
from reduce_dimensionality import reduce_dimensionality
from verify_data import verify_data_files
import sys

def main():
    # Setup paths
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Verify data directory and files
    print("Verifying data files...")
    if not verify_data_files():
        print("❌ Please ensure all required files are present")
        sys.exit(1)
    
    # Process pipeline
    try:
        print("\nStep 1: Normalizing features...")
        normalize_features()
        
        print("\nStep 2: Reducing dimensionality...")
        reduce_dimensionality()
        
        print("\n✅ Music processing pipeline complete!")
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 