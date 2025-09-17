#!/usr/bin/env python3
"""
Convert trained YOLOv11 PyTorch model to ONNX format
"""

import torch
from ultralytics import YOLO
import os
from pathlib import Path

def convert_to_onnx():
    """Convert the trained YOLOv11 model to ONNX format"""

    # Input PyTorch model path
    pt_model_path = "detector/data/BLEARVIS_BestWeights_YOLOv11.pt"

    # Output ONNX model path
    onnx_model_path = "detector/data/BLEARVIS_BestWeights_YOLOv11.onnx"

    # Check if PyTorch model exists
    if not Path(pt_model_path).exists():
        print(f"❌ PyTorch model not found: {pt_model_path}")
        print("💡 Make sure you have trained the model and copied it to the correct location:")
        print("   cp runs/train/blearvis_yolo11/weights/best.pt detector/data/BLEARVIS_BestWeights_YOLOv11.pt")
        return False

    try:
        print(f"🔄 Loading PyTorch model: {pt_model_path}")
        model = YOLO(pt_model_path)
        print("✅ Model loaded successfully")

        # Convert to ONNX
        print(f"🔄 Converting to ONNX: {onnx_model_path}")

        # Export with optimal settings for inference
        model.export(
            format='onnx',
            imgsz=640,  # Input image size (should match training)
            half=False,  # Use FP32 for compatibility
            dynamic=False,  # Fixed input size for better performance
            simplify=True,  # Simplify the model
            opset=17  # ONNX opset version
        )

        # Check if ONNX file was created
        if Path(onnx_model_path).exists():
            file_size = Path(onnx_model_path).stat().st_size / (1024 * 1024)  # Size in MB
            print(f"💾 ONNX model size: {file_size:.2f} MB")
            print("✅ ONNX conversion completed successfully!")
            print(f"📁 ONNX model saved to: {onnx_model_path}")
            return True
        else:
            print("❌ ONNX conversion failed - output file not found")
            return False

    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_onnx_model():
    """Test the converted ONNX model"""
    onnx_model_path = "detector/data/BLEARVIS_BestWeights_YOLOv11.onnx"
    test_image_path = "detector/data/images/P1000286.JPG"

    if not Path(onnx_model_path).exists():
        print(f"❌ ONNX model not found: {onnx_model_path}")
        return False

    try:
        print(f"🔄 Testing ONNX model: {onnx_model_path}")

        # Load ONNX model with YOLO class
        model = YOLO(onnx_model_path)

        # Test with sample image
        if Path(test_image_path).exists():
            print(f"🔄 Running inference on: {test_image_path}")
            results = model(test_image_path, conf=0.5, iou=0.45, verbose=False)

            # Print results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    print(f"✅ Detection successful! Found {len(boxes)} objects")
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = result.names[cls]
                        print(f"  - {class_name}: {conf:.3f}")
                else:
                    print("⚠️  No objects detected in test image")

            print("✅ ONNX model test completed successfully!")
            return True
        else:
            print(f"⚠️  Test image not found: {test_image_path}")
            print("✅ ONNX model loaded successfully (skipping inference test)")
            return True

    except Exception as e:
        print(f"❌ ONNX model test failed: {e}")
        return False

if __name__ == "__main__":
    print("YOLOv11 to ONNX Converter")
    print("=" * 40)

    # Convert model
    if convert_to_onnx():
        print("\n" + "-" * 40)
        # Test the converted model
        test_onnx_model()

        print("\n" + "=" * 40)
        print("🎉 Conversion completed!")
        print("\nNext steps:")
        print("1. Update your config.yml to use YOLO_VERSION: v11")
        print("2. The detector will automatically use the ONNX model for inference")
        print("3. ONNX models typically provide faster inference than PyTorch models")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")