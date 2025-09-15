#!/usr/bin/env python3
"""
YOLO Model Downloader for BLEARVIS
Combines information display and actual model downloading functionality
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def show_model_info():
    """Display information about available YOLO models"""
    print("🤖 YOLO Model Information")
    print("=" * 50)
    print("Available YOLO v11 models:")
    print("  📦 yolo11n.pt - Nano (fastest, ~30-50 FPS, good for real-time)")
    print("  📦 yolo11s.pt - Small (balanced performance)")
    print("  📦 yolo11m.pt - Medium (better accuracy)")
    print("  📦 yolo11l.pt - Large (high accuracy)")
    print("  📦 yolo11x.pt - Extra Large (best accuracy, slowest)")
    print("")
    print("💡 Tips:")
    print("  • Use yolo11n.pt for real-time applications")
    print("  • Use yolo11x.pt for maximum accuracy")
    print("  • Models are downloaded automatically when first used")
    print("  • Downloaded models are cached locally")
    print("")

def download_single_model(model_name, save_path='models/'):
    """
    Download and save a single YOLO model locally

    Args:
        model_name (str): Name of the model to download
        save_path (str): Directory to save the model

    Returns:
        str: Path to downloaded model, or None if failed
    """
    # Create save directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {model_name}...")

    try:
        # Check if model already exists
        model_path = os.path.join(save_path, model_name)
        if os.path.exists(model_path):
            print(f"✅ Model already exists: {model_path}")
            return model_path

        # Download the model using Ultralytics
        model = YOLO(model_name)

        # Save the model locally (optional, since Ultralytics caches automatically)
        model.save(model_path)

        print(f"✅ Model saved to: {model_path}")
        return model_path

    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return None

def download_all_models(save_path='models/'):
    """Download all common YOLO v11 models"""
    print("🚀 Downloading all YOLO v11 models for offline use...")
    print("=" * 60)

    models_to_download = [
        'yolo11n.pt',  # Nano - fastest
        'yolo11s.pt',  # Small
        'yolo11m.pt',  # Medium
        'yolo11l.pt',  # Large
        'yolo11x.pt',  # Extra Large - most accurate
    ]

    downloaded_models = []

    for model_name in models_to_download:
        print(f"\n🔄 Processing {model_name}...")
        model_path = download_single_model(model_name, save_path)
        if model_path:
            downloaded_models.append(model_path)

    print("\n" + "=" * 60)
    print(f"✅ Downloaded {len(downloaded_models)}/{len(models_to_download)} models")

    if downloaded_models:
        print("\n📁 Downloaded models:")
        for model in downloaded_models:
            print(f"  📦 {model}")

        print("\n💡 You can now use these models offline!")
        print("   Update your config.yml or detector to use the local paths.")
    else:
        print("❌ No models were downloaded successfully.")

    return downloaded_models

def test_model_loading(model_name):
    """Test if a model can be loaded successfully"""
    print(f"🧪 Testing {model_name}...")

    try:
        model = YOLO(model_name)
        print(f"✅ {model_name} loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return False

def interactive_menu():
    """Interactive menu for model operations"""
    while True:
        print("\n" + "=" * 50)
        print("🤖 YOLO Model Manager")
        print("=" * 50)
        print("1. 📋 Show model information")
        print("2. 📥 Download single model")
        print("3. 🚀 Download all models")
        print("4. 🧪 Test model loading")
        print("5. 📂 List downloaded models")
        print("6. ❌ Exit")
        print("=" * 50)

        try:
            choice = input("Choose an option (1-6): ").strip()

            if choice == '1':
                show_model_info()

            elif choice == '2':
                model_name = input("Enter model name (e.g., yolo11n.pt): ").strip()
                if model_name:
                    download_single_model(model_name)
                else:
                    print("❌ No model name provided.")

            elif choice == '3':
                download_all_models()

            elif choice == '4':
                model_name = input("Enter model name to test: ").strip()
                if model_name:
                    test_model_loading(model_name)
                else:
                    print("❌ No model name provided.")

            elif choice == '5':
                print("📂 Downloaded models in current directory:")
                model_extensions = ['.pt', '.onnx', '.yaml']
                found_models = []

                for file in os.listdir('.'):
                    if any(file.endswith(ext) for ext in model_extensions):
                        if 'yolo' in file.lower():
                            found_models.append(file)

                if found_models:
                    for model in sorted(found_models):
                        size = os.path.getsize(model) / (1024 * 1024)  # Size in MB
                        print(f"  - {model} ({size:.1f} MB)")
                else:
                    print("❌ No YOLO models found in current directory.")

            elif choice == '6':
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please select 1-6.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function with command-line argument support"""
    if len(sys.argv) > 1:
        # Command-line mode
        command = sys.argv[1].lower()

        if command == 'info':
            show_model_info()

        elif command == 'download' and len(sys.argv) > 2:
            model_name = sys.argv[2]
            download_single_model(model_name)

        elif command == 'download-all':
            download_all_models()

        elif command == 'test' and len(sys.argv) > 2:
            model_name = sys.argv[2]
            test_model_loading(model_name)

        else:
            print("Usage:")
            print("  python download_models.py info")
            print("  python download_models.py download <model_name>")
            print("  python download_models.py download-all")
            print("  python download_models.py test <model_name>")
            print("  python download_models.py (interactive mode)")

    else:
        # Interactive mode
        interactive_menu()

if __name__ == "__main__":
    main()