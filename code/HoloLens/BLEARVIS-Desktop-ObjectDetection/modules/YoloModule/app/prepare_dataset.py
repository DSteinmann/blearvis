#!/usr/bin/env python3
"""
Prepare dataset for YOLOv11 training
Creates train.txt and val.txt files with image paths
"""

import os
import glob
from pathlib import Path

def prepare_dataset():
    dataset_path = Path("detector/data/dataset")
    image_extensions = ['*.jpg', '*.jpeg', '*.png']

    # Collect all image files from subfolders
    all_images = []
    for ext in image_extensions:
        for folder in ['Tractorbot', 'roboticarm_a', 'roboticarm_b']:
            folder_path = dataset_path / folder
            if folder_path.exists():
                images = list(folder_path.glob(ext))
                all_images.extend(images)

    print(f"Found {len(all_images)} images")

    # Split into train/val (80/20)
    train_count = int(len(all_images) * 0.8)
    train_images = all_images[:train_count]
    val_images = all_images[train_count:]

    # Write train.txt
    with open(dataset_path / 'train.txt', 'w') as f:
        for img in train_images:
            f.write(f"{img}\n")

    # Write val.txt
    with open(dataset_path / 'val.txt', 'w') as f:
        for img in val_images:
            f.write(f"{img}\n")

    print(f"Created train.txt with {len(train_images)} images")
    print(f"Created val.txt with {len(val_images)} images")

if __name__ == "__main__":
    prepare_dataset()