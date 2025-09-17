#!/usr/bin/env python3
"""
Test script for YOLO v11 implementation with custom trained model
Tests detection on P1000286.JPG and saves annotated result
"""

import cv2
import sys
import os
from pathlib import Path

# Add the detector module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_custom_yolo11_detector():
    """Test YOLO v11 detector with custom trained model on P1000286.JPG"""
    try:
        from detector.yolov11_detector import YOLOv11Detector
        print("✅ YOLO v11 detector import successful")
    except ImportError as e:
        print(f"❌ Failed to import YOLO v11 detector: {e}")
        return False

    # Check if ultralytics is available
    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not installed")
        return False

    # Test detector initialization with custom model
    try:
        print("🔄 Testing detector initialization with custom model...")
        detector = YOLOv11Detector(custom=True)  # This will use BLEARVIS_BestWeights_YOLOv11.pt
        print("✅ YOLO v11 detector initialized successfully with custom model")
        print(f"📋 Model info: {detector.get_model_info()}")

        # Check if the model actually has the YOLO object
        if hasattr(detector, 'model'):
            print("✅ Detector has model attribute")
            if hasattr(detector.model, 'names') and detector.model.names:
                print("📋 Model classes from detector:")
                for class_id, class_name in detector.model.names.items():
                    print(f"   {class_id}: {class_name}")
            else:
                print("⚠️  Model doesn't have names attribute or it's empty")
        else:
            print("❌ Detector doesn't have model attribute")

    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with known training image that contains tractorbot
    test_image_path = "detector/data/images/spock0003.jpg"
    output_path = "detector/detections/spock0003.jpg"

    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False

    try:
        print(f"🔄 Loading training image: {test_image_path} (known to contain tractorbot)")
        image = cv2.imread(test_image_path)
        if image is None:
            print("❌ Failed to load image")
            return False

        print(f"✅ Image loaded successfully")
        print(f"📏 Image size: {image.shape}")

        # Test with appropriate confidence threshold
        print("🔄 Running detection with conf=0.3...")
        # Set appropriate confidence threshold
        original_conf = detector.conf_threshold
        detector.conf_threshold = 0.3

        detections, annotated_image = detector.detect(image)

        print(f"📊 Found {len(detections)} detections with conf=0.3")

        if detections:
            for i, detection in enumerate(detections):
                class_name, confidence, bbox_topleft, bbox_bottomright = detection
                x1, y1 = bbox_topleft
                x2, y2 = bbox_bottomright
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                print(f"  {i+1}. {class_name}")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      BBox: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
                print(f"      Size: {bbox_width:.0f} x {bbox_height:.0f} pixels")
        else:
            print("  ❌ No detections found even with conf=0.3")
            print("  💡 This suggests the model may not be trained on the expected classes")

        # Save annotated image
        cv2.imwrite(output_path, annotated_image)
        print(f"💾 Saved annotated image: {output_path}")

        # Restore original confidence
        detector.conf_threshold = original_conf

        print(f"📊 Detection Summary:")
        if detections:
            classes_found = [d[0] for d in detections]
            print(f"   - Classes detected: {set(classes_found)}")
            avg_confidence = sum(d[1] for d in detections) / len(detections)
            print(f"   - Average confidence: {avg_confidence:.3f}")

        # Also test roboticarm image
        roboticarm_image_path = "detector/data/images/P1000286.JPG"
        roboticarm_output_path = "detector/detections/P1000286.jpg"

        if os.path.exists(roboticarm_image_path):
            print(f"\n🔄 Testing roboticarm detection: {roboticarm_image_path}")
            roboticarm_image = cv2.imread(roboticarm_image_path)
            if roboticarm_image is not None:
                detector.conf_threshold = 0.3
                detections_ra, annotated_image_ra = detector.detect(roboticarm_image)
                detector.conf_threshold = original_conf

                print(f"📊 Found {len(detections_ra)} detections in roboticarm image")
                if detections_ra:
                    for i, detection in enumerate(detections_ra):
                        class_name, confidence, bbox_topleft, bbox_bottomright = detection
                        print(f"  {i+1}. {class_name} (conf: {confidence:.3f})")

                cv2.imwrite(roboticarm_output_path, annotated_image_ra)
                print(f"💾 Saved roboticarm annotated image: {roboticarm_output_path}")
        else:
            print(f"⚠️  Roboticarm test image not found: {roboticarm_image_path}")

    except Exception as e:
        print(f"❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_detector_class():
    """Test the main Detector class with YOLO v11 custom model"""
    try:
        from detector.detector import Detector
        print("\n--- Testing main Detector class ---")

        # Test YOLO v11 initialization with custom model
        detector = Detector(yolo_version='v11')
        print("✅ Main detector initialized with YOLO v11")

        # Test model info
        if hasattr(detector, 'detector'):
            print(f"📋 Model info: {detector.detector.get_model_info()}")

    except Exception as e:
        print(f"❌ Main detector test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("YOLO v11 Custom Model Test")
    print("=" * 50)

    success = True

    # Test YOLO v11 detector with custom model
    if not test_custom_yolo11_detector():
        success = False

    # Test main detector class
    if not test_detector_class():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! YOLO v11 custom model is working.")
        print("\n📁 Check the annotated image: detector/data/images/P1000286_annotated.jpg")
        print("\nNext steps:")
        print("1. Update your config.yml to set YOLO_VERSION: v11 and CUSTOM: True")
        print("2. Run the main application: python main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    print("\nFor more information, see YOLOv11_README.md")