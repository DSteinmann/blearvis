#!/usr/bin/env python3
"""
Fine-tune YOLOv11 on BLEARVIS custom dataset
"""

from ultralytics import YOLO

def train_yolo11():
    # Load a pre-trained YOLOv11 model
    model = YOLO('yolo11n.pt')  # or yolov11s.pt for better accuracy

    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=100,  # Full training epochs for GPU
        imgsz=640,
        batch=16,  # Larger batch size for GPU training
        name='blearvis_yolo11',
        project='runs/train',
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        cache=False,  # Set to True if you have enough RAM
        device=0,  # Use GPU 0 (change to 'cpu' if no GPU available)
        workers=4,  # Data loading workers for GPU
        patience=50,  # Early stopping patience
        optimizer='Adam',  # Optimizer
        lr0=0.001,  # Initial learning rate
        lrf=0.01,  # Final learning rate fraction
        momentum=0.937,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # Optimizer weight decay
        warmup_epochs=3.0,  # Warmup epochs
        warmup_momentum=0.8,  # Warmup initial momentum
        warmup_bias_lr=0.1,  # Warmup initial bias lr
        box=7.5,  # Box loss gain
        cls=0.5,  # Cls loss gain
        dfl=1.5,  # DFL loss gain
        pose=12.0,  # Pose loss gain
        kobj=1.0,  # Keypoint obj loss gain
        label_smoothing=0.0,  # Label smoothing
        nbs=64,  # Nominal batch size
        hsv_h=0.015,  # Image HSV-Hue augmentation
        hsv_s=0.7,  # Image HSV-Saturation augmentation
        hsv_v=0.4,  # Image HSV-Value augmentation
        degrees=0.0,  # Image rotation (+/- deg)
        translate=0.1,  # Image translation (+/- fraction)
        scale=0.5,  # Image scale (+/- gain)
        shear=0.0,  # Image shear (+/- deg)
        perspective=0.0,  # Image perspective (+/- fraction), range 0-0.001
        flipud=0.0,  # Image flip up-down (probability)
        fliplr=0.5,  # Image flip left-right (probability)
        mosaic=1.0,  # Image mosaic (probability)
        mixup=0.0,  # Image mixup (probability)
        copy_paste=0.0,  # Segment copy-paste (probability)
    )

    # The trained model will be saved in runs/train/blearvis_yolo11/weights/best.pt
    print("Training completed!")
    print("Best model saved as: runs/train/blearvis_yolo11/weights/best.pt")
    print("Copy this file to detector/data/BLEARVIS_BestWeights_YOLOv11.pt for use in the application")

if __name__ == "__main__":
    train_yolo11()