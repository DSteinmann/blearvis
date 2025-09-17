# YOLO v11 Implementation for BLEARVIS

This document describes the YOLO v11 implementation added to the BLEARVIS object detection system.

## Overview

YOLO v11 is the latest version of the YOLO (You Only Look Once) object detection model series from Ultralytics. It offers improved accuracy and performance compared to previous versions.

## Features

- **Dual YOLO Support**: The system now supports both YOLO v7 (original) and YOLO v11
- **Easy Switching**: Switch between YOLO versions via configuration file
- **Backward Compatibility**: Existing YOLO v7 functionality remains unchanged
- **Custom Model Support**: Support for both pre-trained and custom-trained YOLO v11 models

## Installation

### 1. Update Environment

Update your conda environment to include the ultralytics package:

```bash
# If using conda environment
conda env update -f yolov7-env.yaml

# Or install ultralytics directly
pip install ultralytics
```

### 2. Download YOLO v11 Models

Run the download script to get information about available models:

```bash
cd modules/YoloModule/app
python download_yolo11.py
```

Or manually download models from the [Ultralytics repository](https://github.com/ultralytics/ultralytics).

## Configuration

### Basic Configuration

Edit `config.yml` to select YOLO version:

```yaml
# YOLO version selection: 'v7' or 'v11'
YOLO_VERSION: v11
```

### Model Configuration

For YOLO v11, the system will automatically:
- Use `detector/data/BLEARVIS_BestWeights_YOLOv11.pt` for custom models
- Fall back to `yolov11n.pt` (default nano model) for pre-trained models

### Model Format Selection

The YOLO v11 detector automatically selects the best available model format:

1. **ONNX** (`.onnx`) - Preferred for production (faster inference)
2. **PyTorch** (`.pt`) - Used when ONNX is not available

**Automatic Detection:**
- If `BLEARVIS_BestWeights_YOLOv11.onnx` exists → Uses ONNX
- Else if `BLEARVIS_BestWeights_YOLOv11.pt` exists → Uses PyTorch
- Otherwise → Error with training instructions

## Usage

### Switching Between YOLO Versions

1. **YOLO v7** (original):
   ```yaml
   YOLO_VERSION: v7
   ```

2. **YOLO v11**:
   ```yaml
   YOLO_VERSION: v11
   ```

### Running the Application

The application will automatically detect the YOLO version from the configuration and initialize the appropriate detector:

```bash
# From repository root
cd modules/YoloModule/app
python main.py
```

## Model Options

### Pre-trained Models

YOLO v11 offers several pre-trained models with different performance characteristics:

- **yolov11n.pt**: Nano (fastest, ~40 FPS, good for real-time)
- **yolov11s.pt**: Small (balanced performance)
- **yolov11m.pt**: Medium (better accuracy)
- **yolov11l.pt**: Large (high accuracy)
- **yolov11x.pt**: Extra Large (highest accuracy, slowest)

### Custom Models

For custom-trained models:
1. Train your model using Ultralytics
2. Save the model as `.pt` file
3. Place it in `detector/data/` directory
4. Name it `BLEARVIS_BestWeights_YOLOv11.pt`
5. Set `CUSTOM: True` in config.yml

## Training Custom YOLO v11 Models

### 1. Prepare the Dataset

Run the dataset preparation script:

```bash
cd modules/YoloModule/app
python prepare_dataset.py
```

This creates `train.txt` and `val.txt` files with image paths.

### 2. Train the Model

Run the training script:

```bash
python train_yolo11.py
```

The script will:
- Load a pre-trained YOLOv11n model
- Fine-tune it on your custom dataset
- Save checkpoints and the best model

### 3. Test the Model

After training completes, test your model:

```bash
python test_yolo11.py
```

This will:
- Load your trained `BLEARVIS_BestWeights_YOLOv11.pt` model
- Run detection on `P1000286.JPG`
- Save the annotated result as `P1000286_annotated.jpg`
- Display detection results and statistics

### 4. Convert to ONNX (Optional but Recommended)

For better inference performance, convert the trained PyTorch model to ONNX format:

```bash
python convert_to_onnx.py
```

This will:
- Load your trained `BLEARVIS_BestWeights_YOLOv11.pt` model
- Convert it to `BLEARVIS_BestWeights_YOLOv11.onnx`
- Test the converted model with a sample image
- ONNX models typically provide faster inference than PyTorch models

### 5. Use the Trained Model

After training and optional ONNX conversion:
1. Copy the best model to the expected location:
   ```bash
   cp runs/train/blearvis_yolo11/weights/best.pt detector/data/BLEARVIS_BestWeights_YOLOv11.pt
   ```

2. Update `config.yml`:
   ```yaml
   YOLO_VERSION: v11
   CUSTOM: True
   ```

3. Run the application:
   ```bash
   python main.py
   ```

### ONNX Conversion Benefits

Converting to ONNX format provides several advantages:

- **Faster Inference**: ONNX models are optimized for inference
- **Cross-Platform**: Can run on different frameworks (ONNX Runtime, TensorRT, etc.)
- **Smaller Runtime**: No need to install PyTorch for inference
- **Production Ready**: Better suited for deployment

**When to use ONNX:**
- For production deployment
- When you need maximum inference speed
- For edge devices with limited resources
- When PyTorch is not available in the target environment

**When to use PyTorch (.pt):**
- During development and testing
- When you need the latest YOLOv11 features
- For fine-tuning or further training

### Training Parameters

The training script uses optimized hyperparameters for small datasets:
- **Epochs**: 100 (adjust based on convergence)
- **Image Size**: 640x640
- **Batch Size**: 16 (reduce if GPU memory is limited)
- **Learning Rate**: 0.001 with warmup
- **Data Augmentation**: Moderate augmentation for robustness

### Monitoring Training

Training progress is logged to the console and saved in `runs/train/blearvis_yolo11/`. Use TensorBoard to monitor:

```bash
tensorboard --logdir runs/train
```

### Expected Results

With ~500 training images across 2 classes, expect:
- Training time: 1-2 hours on a modern GPU
- Final mAP: 0.8-0.95 depending on data quality
- Model size: ~5-20MB depending on base model used

### CPU Training Notes

If training on CPU (no GPU available):

- The script automatically detects and uses CPU
- Training will be significantly slower (hours instead of minutes)
- Consider reducing epochs to 50-100 initially
- Use smaller batch sizes (4-8)
- Set workers=0 to avoid multiprocessing issues
- Be patient - CPU training can take several hours

### GPU Training (Recommended)

For faster training, use a GPU:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check GPU availability
python check_gpu.py

# Then run training
python train_yolo11.py
```

**GPU Utilization Issues:**

If GPU usage is low during training:

1. **Check CUDA status**: Run `python check_gpu.py`
2. **Verify PyTorch CUDA**: Ensure you have `torch==2.8.0+cu129` or similar
3. **Batch size**: Increase batch size (16-32) for better GPU utilization
4. **Data loading**: Ensure `workers > 0` for parallel data loading
5. **Monitor GPU**: Use `nvidia-smi` or run `python monitor_gpu.py` in another terminal during training
6. **CUDA version match**: Ensure CUDA toolkit version matches PyTorch CUDA version

**GPU Monitoring Script:**

Run this in a separate terminal during training:

```bash
python monitor_gpu.py
```

**Expected GPU usage**: 80-100% during training with proper batch size

## Performance Comparison

| Model | YOLO v7 | YOLO v11 |
|-------|---------|----------|
| Accuracy | Good | Better |
| Speed | Fast | Similar/Faster |
| Model Size | Moderate | Optimized |
| Ease of Use | Complex setup | Simple |

## Troubleshooting

### Import Errors

If you see import errors for ultralytics:
```bash
pip install ultralytics
```

### Model Loading Errors

- Ensure the model file exists in the correct location
- Check that the model file is not corrupted
- Verify the model format (.pt for PyTorch, .onnx for ONNX)

### Performance Issues

- Try smaller models (yolov11n.pt) for better performance
- Adjust confidence threshold in config.yml
- Consider using GPU acceleration

### Dataset Structure Fix

**Important**: The initial training may have failed to use annotations properly due to incorrect dataset structure. If you encounter detection issues:

1. **Restructure the dataset**:
   ```bash
   python restructure_dataset.py
   ```

2. **Verify the data.yaml** uses directory paths:
   ```yaml
   train: train/
   val: val/
   ```

3. **Retrain the model** with the corrected structure:
   ```bash
   python train_yolo11.py
   ```

This ensures annotations are properly paired with images during training.

## API Changes

The detection API remains the same for both YOLO versions:

```python
detections, annotated_frame = detector.detect(frame)
```

Where:
- `detections`: List of tuples `(class_name, confidence, bbox_topleft, bbox_bottomright)`
- `annotated_frame`: Original frame with bounding boxes drawn

## Future Enhancements

- ONNX export support for YOLO v11
- TensorRT optimization
- Multi-model ensemble support
- Automatic model selection based on hardware

## Support

For issues specific to YOLO v11 implementation, check:
1. [Ultralytics Documentation](https://docs.ultralytics.com/)
2. [YOLO v11 GitHub](https://github.com/ultralytics/ultralytics)
3. Existing YOLO v7 documentation in this project