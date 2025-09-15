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