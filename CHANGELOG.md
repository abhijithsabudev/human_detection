## 1.0.0

### Initial Release 🎉

**Features:**
* Human detection from image file path
* Human detection from image bytes
* Configurable confidence threshold
* GPU acceleration support (Android & iOS)
* Configurable thread count for CPU inference
* Custom model support
* Detailed detection results with:
  - Confidence scores
  - Processing time
  - Bounding box coordinates
* Comprehensive example application

### Supported Platforms

* ✅ Android (API 24+)
* ✅ iOS (13.0+)

### Model

* **Pre-trained SSD MobileNet V1** from TensorFlow Hub
* Trained on COCO dataset (detects "person" class)
* Optimized for mobile deployment with TensorFlow Lite
* Model size: ~4 MB
* Input: 300x300 RGB images
* Supports both object detection and binary classifier models
