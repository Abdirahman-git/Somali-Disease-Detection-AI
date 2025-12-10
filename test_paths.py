from pathlib import Path
import sys

print("Testing project structure...")
print("="*50)

# Check current directory
current_dir = Path.cwd()
print(f"Current directory: {current_dir}")

# List all files and folders
print("\nProject structure:")
for item in current_dir.iterdir():
    if item.is_dir():
        print(f"📁 {item.name}/")
        # List contents of important directories
        if item.name in ['data', 'notebooks', 'models', 'app']:
            for subitem in item.iterdir():
                print(f"    📄 {subitem.name}")
    else:
        print(f"📄 {item.name}")

print("\n" + "="*50)

# Test config import
try:
    from config import DATA_FILE, MODELS_DIR, DATA_DIR
    print(f"✅ Config loaded successfully!")
    print(f"   Data file: {DATA_FILE}")
    print(f"   Data exists: {DATA_FILE.exists()}")
    print(f"   Models dir: {MODELS_DIR}")
    print(f"   Data dir: {DATA_DIR}")
except ImportError as e:
    print(f"❌ Error importing config: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")

print("\n" + "="*50)
print("If you see ✅ above, you're ready to train the model!")