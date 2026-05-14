# Human Detection

[![pub package](https://img.shields.io/pub/v/human_detection.svg)](https://pub.dev/packages/human_detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-android%20%7C%20ios-green.svg)](https://flutter.dev)

A lightweight Flutter plugin for detecting humans in images using machine learning. Uses a pre-trained SSD MobileNet model with TensorFlow Lite for efficient on-device inference.

## Features

- 🚀 **Zero Setup** - Just call `HumanDetection.detect()` - no initialization needed!
- 🔍 **Human Detection** - Detect if an image contains a human with high accuracy
- 📦 **Pre-trained Model** - Uses Google's SSD MobileNet V1 trained on COCO dataset
- ⚡ **Non-blocking** - Runs asynchronously without blocking the UI
- 📱 **Cross-platform** - Works on both Android and iOS
- 🎯 **GPU Acceleration** - Optional GPU delegate for faster processing
- 📊 **Detailed Results** - Get confidence scores, processing time, and bounding boxes

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

### Basic Usage - Just One Line!

```dart
import 'package:human_detection/human_detection.dart';

// That's it! No initialization required.
final result = await HumanDetection.detect('/path/to/image.jpg');

print('Is human: ${result.isHuman}');
print('Confidence: ${(result.confidence * 100).toStringAsFixed(1)}%');
print('Processing time: ${result.processingTimeMs}ms');
```

### Detection from Bytes

```dart
import 'dart:io';

final bytes = await File('image.jpg').readAsBytes();
final result = await HumanDetection.detectFromBytes(bytes);
```

### Custom Configuration (Optional)

```dart
// Configure once if you need custom settings
await HumanDetection.configure(HumanDetectionOptions(
  confidenceThreshold: 0.7,  // Higher threshold for stricter detection
  useGpuDelegate: true,       // Enable GPU acceleration (disabled by default)
  numThreads: 4,              // Number of CPU threads
));

// Then detect as usual
final result = await HumanDetection.detect('/path/to/image.jpg');
```

### Clean Up (Optional)

```dart
// Call dispose when completely done to free memory
// The next detect call will automatically re-initialize
await HumanDetection.dispose();
```

### Complete Example

```dart
import 'package:flutter/material.dart';
import 'package:human_detection/human_detection.dart';
import 'package:image_picker/image_picker.dart';

class HumanDetectionExample extends StatefulWidget {
  @override
  State<HumanDetectionExample> createState() => _HumanDetectionExampleState();
}

class _HumanDetectionExampleState extends State<HumanDetectionExample> {
  HumanDetectionResult? _result;
  bool _isLoading = false;

  Future<void> _pickAndDetect() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.gallery);
    
    if (image != null) {
      setState(() => _isLoading = true);
      
      // Just one line - no setup needed!
      final result = await HumanDetection.detect(image.path);
      
      setState(() {
        _result = result;
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_isLoading)
              const CircularProgressIndicator()
            else if (_result != null) ...[
              Icon(
                _result!.isHuman ? Icons.person : Icons.person_off,
                size: 64,
                color: _result!.isHuman ? Colors.green : Colors.red,
              ),
              Text('Confidence: ${(_result!.confidence * 100).toStringAsFixed(1)}%'),
            ],
            ElevatedButton(
              onPressed: _isLoading ? null : _pickAndDetect,
              child: const Text('Select Image'),
            ),
          ],
        ),
      ),
    );
  }
}
```

## API Reference

### HumanDetection (Static Methods)

| Method | Description |
|--------|-------------|
| `detect(imagePath, {options})` | Detect human from file path (auto-initializes) |
| `detectFromBytes(bytes, {options})` | Detect human from image bytes (auto-initializes) |
| `configure(options)` | Pre-configure detection options |
| `dispose()` | Release resources (optional) |
| `isInitialized` | Check if model is loaded |

### HumanDetectionOptions

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `confidenceThreshold` | `double` | `0.5` | Minimum confidence to consider detection positive |
| `useGpuDelegate` | `bool` | `false` | Use GPU acceleration if available |
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
