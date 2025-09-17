'''
YOLO v11 Object Detection Implementation
Using Ultralytics YOLO v11 for object detection
'''

import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path

class YOLOv11Detector:
    def __init__(self, model_path=None, conf_threshold=0.3, iou_threshold=0.45, custom=False):
        """
        Initialize YOLO v11 detector

        Args:
            model_path (str): Path to the YOLO v11 model file (.pt or .onnx)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            custom (bool): Whether using custom trained model
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.custom = custom

        # Default model path if not provided
        if model_path is None:
            if custom:
                # Prefer ONNX for better performance, fall back to PyTorch
                onnx_path = "detector/data/BLEARVIS_BestWeights_YOLOv11.onnx"
                pt_path = "detector/data/BLEARVIS_BestWeights_YOLOv11.pt"

                if Path(onnx_path).exists():
                    model_path = onnx_path
                    print("📁 Using ONNX model for better performance")
                elif Path(pt_path).exists():
                    model_path = pt_path
                    print("📁 Using PyTorch model")
                else:
                    print("❌ No custom model found. Please train a model first.")
                    print("   Run: python train_yolo11.py")
                    print("   Then: python convert_to_onnx.py (optional)")
                    raise FileNotFoundError(f"Neither {onnx_path} nor {pt_path} found")
            else:
                # Use YOLO v11 models with correct naming (without 'v')
                model_path = "../yolo11n.pt"  # Default YOLO v11 nano model

        self.model_path = model_path

        # Initialize the model
        try:
            # Try to load the specified model path first
            if Path(model_path).exists():
                self.model = YOLO(model_path)
                print(f"✅ Loaded YOLO model: {model_path}")
            else:
                # Model doesn't exist locally, let Ultralytics handle download
                print(f"📥 Model {model_path} not found locally, downloading...")
                self.model = YOLO(model_path)  # Ultralytics will auto-download
                print(f"✅ Downloaded and loaded YOLO model: {model_path}")
        except Exception as e:
            print(f"❌ Error loading YOLO model '{model_path}': {e}")
            # Try fallback models
            fallback_models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11n', 'yolo11s', 'yolov8n.pt', 'yolov8s.pt']
            for fallback in fallback_models:
                try:
                    print(f"🔄 Attempting fallback with {fallback}...")
                    self.model = YOLO(fallback)
                    print(f"✅ Successfully loaded fallback model: {fallback}")
                    self.model_path = fallback
                    break
                except Exception as e2:
                    print(f"❌ Fallback {fallback} failed: {str(e2)[:50]}...")
                    continue
            else:
                raise Exception("Could not load any YOLO model. Please check your internet connection and try again.")

    def detect(self, frame):
        """
        Perform object detection on a frame

        Args:
            frame: Input image frame (BGR format)

        Returns:
            detections: List of detections [(class_name, confidence, bbox_topleft, bbox_bottomright), ...]
            annotated_frame: Frame with detections drawn
        """
        # Convert BGR to RGB for YOLO v11
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model(rgb_frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

        detections = []
        annotated_frame = frame.copy()

        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # Get class name
                    class_name = result.names[class_id]

                    # Convert to integer coordinates
                    bbox_topleft = (int(x1), int(y1))
                    bbox_bottomright = (int(x2), int(y2))

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, bbox_topleft, bbox_bottomright, (0, 255, 0), 2)

                    # Draw label
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Add to detections list
                    detections.append((class_name, float(confidence), bbox_topleft, bbox_bottomright))

        return detections, annotated_frame

    def get_model_info(self):
        """
        Get information about the loaded model

        Returns:
            dict: Model information
        """
        return {
            'model_path': self.model_path,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'custom': self.custom,
            'model_type': 'YOLO v11/v8',
            'note': 'Using YOLO v11 if available, otherwise YOLO v8 fallback'
        }