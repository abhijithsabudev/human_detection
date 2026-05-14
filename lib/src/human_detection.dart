import 'dart:typed_data';

import '../human_detection_platform_interface.dart';
import 'human_detection_options.dart';
import 'human_detection_result.dart';

/// A Flutter plugin for detecting humans in images.
///
/// This plugin uses a machine learning model (MobileNetV2-based) to determine
/// whether an image contains a human or not.
///
/// ## Usage
///
/// ```dart
/// final humanDetection = HumanDetection();
///
/// // Initialize the detector
/// await humanDetection.initialize();
///
/// // Detect human from file path
/// final result = await humanDetection.detectHuman('/path/to/image.jpg');
/// print('Is human: ${result.isHuman}');
/// print('Confidence: ${result.confidence}');
///
/// // Or from bytes
/// final bytes = await File('image.jpg').readAsBytes();
/// final result = await humanDetection.detectHumanFromBytes(bytes);
///
/// // Dispose when done
/// humanDetection.dispose();
/// ```
class HumanDetection {
  bool _isInitialized = false;

  /// Returns true if the detector has been initialized.
  bool get isInitialized => _isInitialized;

  /// Initialize the human detection model.
  ///
  /// This must be called before any detection methods.
  /// The [options] parameter allows customizing the model configuration.
  ///
  /// Throws [HumanDetectionException] if initialization fails.
  Future<void> initialize([HumanDetectionOptions? options]) async {
    if (_isInitialized) return;

    await HumanDetectionPlatform.instance.initialize(
      options ?? HumanDetectionOptions(),
    );
    _isInitialized = true;
  }

  /// Detect if the image at [imagePath] contains a human.
  ///
  /// Returns a [HumanDetectionResult] containing:
  /// - [isHuman]: Whether a human was detected
  /// - [confidence]: Confidence score (0.0 to 1.0)
  /// - [processingTimeMs]: Processing time in milliseconds
  ///
  /// Throws [HumanDetectionException] if detection fails.
  Future<HumanDetectionResult> detectHuman(String imagePath) async {
    _checkInitialized();
    return HumanDetectionPlatform.instance.detectHuman(imagePath);
  }

  /// Detect if the image bytes contain a human.
  ///
  /// This is useful when you have the image data in memory,
  /// for example from a camera capture or network request.
  ///
  /// Returns a [HumanDetectionResult] containing detection results.
  ///
  /// Throws [HumanDetectionException] if detection fails.
  Future<HumanDetectionResult> detectHumanFromBytes(
    Uint8List imageBytes,
  ) async {
    _checkInitialized();
    return HumanDetectionPlatform.instance.detectHumanFromBytes(imageBytes);
  }

  /// Get the current platform version.
  ///
  /// Returns the platform-specific version string.
  Future<String?> getPlatformVersion() {
    return HumanDetectionPlatform.instance.getPlatformVersion();
  }

  /// Dispose of the detector and release resources.
  ///
  /// Call this when you're done using the detector to free memory.
  Future<void> dispose() async {
    if (!_isInitialized) return;

    await HumanDetectionPlatform.instance.dispose();
    _isInitialized = false;
  }

  void _checkInitialized() {
    if (!_isInitialized) {
      throw StateError(
        'HumanDetection has not been initialized. '
        'Call initialize() before using detection methods.',
      );
    }
  }
}
