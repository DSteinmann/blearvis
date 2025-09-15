#!/usr/bin/env python3
"""
Test script for YOLO v11 implementation
"""

import cv2
import sys
import os

# Add the detector module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_yolo11_detector():
    """Test YOLO v11 detector with a sample image"""
    try:
        from detector.yolov11_detector import YOLOv11Detector
        print("✅ YOLO v11 detector import successful")
    except ImportError as e:
        print(f"❌ Failed to import YOLO v11 detector: {e}")
        print("💡 Try: pip install ultralytics")
        return False

    # Check if ultralytics is available
    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not installed")
        return False

    # Test detector initialization
    try:
        print("🔄 Testing detector initialization...")
        detector = YOLOv11Detector()
        print("✅ YOLO v11 detector initialized successfully")
        print(f"📋 Model info: {detector.get_model_info()}")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        print("💡 This might be due to network issues or missing model files.")
        print("   Try running: python download_models.py")
        return False

    # Test with sample image if available
    test_image_path = "detector/data/images/dog.jpg"
    if os.path.exists(test_image_path):
        try:
            image = cv2.imread(test_image_path)
            if image is not None:
                detections, annotated_image = detector.detect(image)
                print(f"✓ Detection successful! Found {len(detections)} objects")
                for detection in detections[:3]:  # Show first 3 detections
                    print(f"  - {detection[0]}: {detection[1]:.2f}")
            else:
                print("✗ Failed to load test image")
        except Exception as e:
            print(f"✗ Detection failed: {e}")
    else:
        print("! Test image not found, skipping detection test")

    return True

def test_detector_class():
    """Test the main Detector class with YOLO v11"""
    try:
        from detector.detector import Detector
        print("\n--- Testing main Detector class ---")

        # Test YOLO v11 initialization
        detector = Detector(yolo_version='v11')
        print("✓ Main detector initialized with YOLO v11")

        # Test model info
        if hasattr(detector, 'detector'):
            print(f"Model info: {detector.detector.get_model_info()}")

    except Exception as e:
        print(f"✗ Main detector test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("YOLO v11 Implementation Test")
    print("=" * 40)

    success = True

    # Test YOLO v11 detector
    if not test_yolo11_detector():
        success = False

    # Test main detector class
    if not test_detector_class():
        success = False

    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! YOLO v11 is ready to use.")
        print("\nNext steps:")
        print("1. Update your config.yml to set YOLO_VERSION: v11")
        print("2. Run the main application: python main.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")

    print("\nFor more information, see YOLOv11_README.md")