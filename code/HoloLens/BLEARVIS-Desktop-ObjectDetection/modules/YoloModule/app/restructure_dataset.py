#!/usr/bin/env python3
"""
Restructure dataset for proper YOLO training format
Creates train/ and val/ directories with images and annotations
"""

import os
import shutil
from pathlib import Path

def restructure_dataset():
    """Move images and annotations to proper train/val directories"""

    dataset_path = Path("detector/data/dataset")
    train_list = dataset_path / "train.txt"
    val_list = dataset_path / "val.txt"
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"

    # Ensure directories exist
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    print("🔄 Restructuring dataset for YOLO training...")

    # Process train.txt
    if train_list.exists():
        print("📋 Processing train.txt...")
        with open(train_list, 'r') as f:
            train_files = [line.strip() for line in f.readlines()]

        for file_path in train_files:
            # Convert to Path object
            src_path = Path(file_path)

            if src_path.exists():
                # Copy image
                dst_image = train_dir / src_path.name
                shutil.copy2(src_path, dst_image)

                # Copy annotation if it exists
                annotation_src = src_path.with_suffix('.txt')
                if annotation_src.exists():
                    annotation_dst = train_dir / annotation_src.name
                    shutil.copy2(annotation_src, annotation_dst)
                else:
                    print(f"⚠️  Missing annotation for: {src_path}")

    # Process val.txt
    if val_list.exists():
        print("📋 Processing val.txt...")
        with open(val_list, 'r') as f:
            val_files = [line.strip() for line in f.readlines()]

        for file_path in val_files:
            # Convert to Path object
            src_path = Path(file_path)

            if src_path.exists():
                # Copy image
                dst_image = val_dir / src_path.name
                shutil.copy2(src_path, dst_image)

                # Copy annotation if it exists
                annotation_src = src_path.with_suffix('.txt')
                if annotation_src.exists():
                    annotation_dst = val_dir / annotation_src.name
                    shutil.copy2(annotation_src, annotation_dst)
                else:
                    print(f"⚠️  Missing annotation for: {src_path}")

    # Count files
    train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.png"))
    val_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.jpeg")) + list(val_dir.glob("*.png"))

    print("✅ Dataset restructured!")
    print(f"📊 Train images: {len(train_images)}")
    print(f"📊 Val images: {len(val_images)}")

    # Verify annotations exist
    train_annotations = list(train_dir.glob("*.txt"))
    val_annotations = list(val_dir.glob("*.txt"))

    print(f"📝 Train annotations: {len(train_annotations)}")
    print(f"📝 Val annotations: {len(val_annotations)}")

    if len(train_images) != len(train_annotations):
        print(f"⚠️  Mismatch! {len(train_images)} train images but {len(train_annotations)} annotations")
    if len(val_images) != len(val_annotations):
        print(f"⚠️  Mismatch! {len(val_images)} val images but {len(val_annotations)} annotations")

if __name__ == "__main__":
    restructure_dataset()