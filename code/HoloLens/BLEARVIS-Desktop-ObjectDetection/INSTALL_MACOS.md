# BLEARVIS YOLO v11 Setup Guide for macOS ARM64

## 🚀 Quick Start

### 1. Set up the Environment
```bash
# Run setup script from repository root
./setup_macos.sh
```

### 2. Activate the Environment
```bash
conda activate blearvis-env
```

### 3. Test YOLO v11
```bash
# From repository root
python modules/YoloModule/app/test_yolo11.py
```

### 4. Run the Application
```bash
# Navigate to app directory
cd modules/YoloModule/app
python main.py
```

## 📋 What's Included

- ✅ **YOLO v11 Support**: Latest Ultralytics YOLO v11 models
- ✅ **Backward Compatibility**: Still supports YOLO v7
- ✅ **macOS ARM64 Optimized**: Native Apple Silicon support
- ✅ **Easy Switching**: Change YOLO version in `config.yml`

## 🔧 Configuration

Edit `modules/YoloModule/app/config.yml`:
```yaml
# Switch between YOLO versions
YOLO_VERSION: v11  # or 'v7' for original
```

## 🧪 Testing

The test script verifies:
- ✅ YOLO v11 installation
- ✅ Model loading
- ✅ Basic detection functionality
- ✅ Configuration loading

## 📚 Documentation

- `YOLOv11_README.md` - Detailed YOLO v11 documentation
- `setup_macos.sh` - Environment setup script
- `test_yolo11.py` - Comprehensive test suite

## 🆘 Troubleshooting

### If setup fails:
```bash
# Clean up and retry
conda env remove -n blearvis-env
./setup_macos.sh
```

### If tests fail:
```bash
# Install missing packages
pip install ultralytics opencv-python
```

### For custom models:
- Place your trained YOLO v11 `.pt` files in `modules/YoloModule/app/detector/data/`
- Update `config.yml` with `CUSTOM: True`

## 🎯 Performance Tips

- **Real-time**: Use `yolov11n.pt` (nano model)
- **Accuracy**: Use `yolov11x.pt` (extra large model)
- **Balance**: Use `yolov11m.pt` (medium model)

Enjoy your upgraded BLEARVIS system with YOLO v11! 🚀