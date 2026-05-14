# Human Detection

[![pub package](https://img.shields.io/pub/v/human_detection.svg)](https://pub.dev/packages/human_detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-android%20%7C%20ios-green.svg)](https://flutter.dev)

A Flutter plugin for detecting humans in images using machine learning. Uses a pre-trained SSD MobileNet model with TensorFlow Lite for efficient on-device inference.

## Features

- 🔍 **Human Detection** - Detect if an image contains a human with high accuracy
- 📦 **Pre-trained Model** - Uses Google's SSD MobileNet V1 trained on COCO dataset
- ⚡ **Fast Inference** - Optimized for real-time detection (~4MB model)
- 📱 **Cross-platform** - Works on both Android and iOS
- 🎯 **GPU Acceleration** - Optional GPU delegate for faster processing
- 🔧 **Configurable** - Adjustable confidence threshold and thread count
- 📊 **Detailed Results** - Get confidence scores, processing time, and bounding boxes
- 🎁 **No Setup Required** - Model is bundled with the package

## Installation

Add this to your package's `pubspec.yaml` file:

```yaml
dependencies:
  human_detection: ^1.0.0
```

Then run:

```bash
flutter pub get
```

### Platform Setup

#### Android

Add the following to your `android/app/build.gradle`:

```gradle
android {
    defaultConfig {
        minSdkVersion 24  // TensorFlow Lite requires API 24+
    }
}
```

#### iOS

Add the following to your `ios/Podfile`:

```ruby
platform :ios, '13.0'
```

## Usage

### Basic Usage

```dart
import 'package:human_detection/human_detection.dart';

// Create an instance
final humanDetection = HumanDetection();

// Initialize the detector
await humanDetection.initialize();

// Detect human from file path
final result = await humanDetection.detectHuman('/path/to/image.jpg');

print('Is human: ${result.isHuman}');
print('Confidence: ${(result.confidence * 100).toStringAsFixed(1)}%');
print('Processing time: ${result.processingTimeMs}ms');

// Don't forget to dispose when done
await humanDetection.dispose();
```

### Detection from Bytes

```dart
import 'dart:io';

final bytes = await File('image.jpg').readAsBytes();
final result = await humanDetection.detectHumanFromBytes(bytes);
```

### Custom Configuration

```dart
await humanDetection.initialize(
  HumanDetectionOptions(
    confidenceThreshold: 0.7,  // Higher threshold for stricter detection
    useGpuDelegate: true,       // Use GPU acceleration
    numThreads: 4,              // Number of CPU threads
  ),
);
```

### Complete Example

```dart
import 'package:flutter/material.dart';
import 'package:human_detection/human_detection.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

class HumanDetectionExample extends StatefulWidget {
  @override
  State<HumanDetectionExample> createState() => _HumanDetectionExampleState();
}

class _HumanDetectionExampleState extends State<HumanDetectionExample> {
  final _humanDetection = HumanDetection();
  HumanDetectionResult? _result;
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _initializeDetector();
  }

  Future<void> _initializeDetector() async {
    try {
      await _humanDetection.initialize();
      setState(() => _isInitialized = true);
    } catch (e) {
      print('Initialization failed: $e');
    }
  }

  Future<void> _pickAndDetect() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.gallery);
    
    if (image != null) {
      final result = await _humanDetection.detectHuman(image.path);
      setState(() => _result = result);
    }
  }

  @override
  void dispose() {
    _humanDetection.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_result != null) ...[
              Icon(
                _result!.isHuman ? Icons.person : Icons.person_off,
                size: 64,
                color: _result!.isHuman ? Colors.green : Colors.red,
              ),
              Text('Confidence: ${(_result!.confidence * 100).toStringAsFixed(1)}%'),
            ],
            ElevatedButton(
              onPressed: _isInitialized ? _pickAndDetect : null,
              child: Text('Select Image'),
            ),
          ],
        ),
      ),
    );
  }
}
```

## API Reference

### HumanDetection

| Method | Description |
|--------|-------------|
| `initialize([options])` | Initialize the detection model |
| `detectHuman(imagePath)` | Detect human from file path |
| `detectHumanFromBytes(bytes)` | Detect human from image bytes |
| `dispose()` | Release resources |
| `isInitialized` | Check if model is ready |

### HumanDetectionOptions

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `confidenceThreshold` | `double` | `0.5` | Minimum confidence to consider detection positive |
| `useGpuDelegate` | `bool` | `true` | Use GPU acceleration if available |
| `numThreads` | `int` | `4` | Number of CPU threads for inference |
| `modelPath` | `String?` | `null` | Custom model path (uses bundled model if null) |

### HumanDetectionResult

| Property | Type | Description |
|----------|------|-------------|
| `isHuman` | `bool` | Whether a human was detected |
| `confidence` | `double` | Confidence score (0.0 to 1.0) |
| `processingTimeMs` | `int?` | Processing time in milliseconds |
| `boundingBox` | `Map<String, double>?` | Bounding box coordinates (top, left, bottom, right) |

## Model Information

This package uses a **pre-trained SSD MobileNet V1** model from TensorFlow Hub:

- **Dataset**: COCO (Common Objects in Context)
- **Detection**: Filters for "person" class (class ID 0)
- **Input Size**: 300x300 RGB
- **Model Size**: ~4 MB
- **Output**: Bounding boxes, confidence scores

The model is bundled with the package - no additional setup required.

### Using a Custom Model

You can use your own TensorFlow Lite model:

```dart
await humanDetection.initialize(
  HumanDetectionOptions(
    modelPath: '/path/to/your/model.tflite',
  ),
);
```

Supported model formats:
- **Object Detection**: Models with 4 outputs (boxes, classes, scores, num_detections)
- **Binary Classifier**: Models with single output (human probability)

## Troubleshooting

### Model Not Found Error

Make sure the TFLite model is included in the package assets. If using a custom model, provide the full path in `HumanDetectionOptions.modelPath`.

### GPU Delegate Errors

If GPU acceleration fails, the plugin automatically falls back to CPU. You can disable GPU explicitly:

```dart
await humanDetection.initialize(
  HumanDetectionOptions(useGpuDelegate: false),
);
```

### iOS Simulator Issues

TensorFlow Lite has limited support on iOS simulators. Test on a physical device for best results.

## Performance Tips

1. **Image Size**: Resize large images before detection to improve speed
2. **GPU Acceleration**: Enable GPU delegate for 2-5x faster inference on supported devices
3. **Thread Count**: Adjust `numThreads` based on device capabilities
4. **Batch Processing**: Reuse the detector instance for multiple images

## Contributing

Contributions are welcome! Please read our [contributing guidelines](https://github.com/abhijithsabudev/human_detection/blob/main/CONTRIBUTING.md) before submitting a pull request.

## Author

**Abhijith K Sabu**
- GitHub: [@abhijithsabudev](https://github.com/abhijithsabudev)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TensorFlow Lite](https://www.tensorflow.org/lite) - On-device ML framework
- [SSD MobileNet](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2) - Pre-trained object detection model
- [COCO Dataset](https://cocodataset.org/) - Training dataset for the model
