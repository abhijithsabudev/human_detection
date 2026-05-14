import 'dart:async';
import 'dart:typed_data';

import '../human_detection_platform_interface.dart';
import 'human_detection_options.dart';
import 'human_detection_result.dart';

/// A lightweight Flutter plugin for detecting humans in images.
///
/// **No initialization required!** Just call the static methods directly:
///
/// ```dart
/// // Detect from file path
/// final result = await HumanDetection.detect('/path/to/image.jpg');
/// print('Is human: ${result.isHuman}');
/// print('Confidence: ${result.confidence}');
///
/// // Detect from bytes
/// final bytes = await File('image.jpg').readAsBytes();
/// final result = await HumanDetection.detectFromBytes(bytes);
/// ```
///
/// The model is automatically initialized on first use and runs
/// asynchronously without blocking the UI.
class HumanDetection {
  // Private constructor - use static methods
  HumanDetection._();

  static bool _initialized = false;
  static Completer<void>? _initCompleter;
  static HumanDetectionOptions? _currentOptions;

  /// Whether the detector has been initialized.
  static bool get isInitialized => _initialized;

  /// Detect if an image file contains a human.
  ///
  /// Example:
  /// ```dart
  /// final result = await HumanDetection.detect('/path/to/image.jpg');
  /// if (result.isHuman) {
  ///   print('Human detected with ${(result.confidence * 100).toStringAsFixed(1)}% confidence');
  /// }
  /// ```
  ///
  /// The model is automatically initialized on first call.
  /// Optionally pass [options] to customize detection behavior.
  static Future<HumanDetectionResult> detect(
    String imagePath, {
    HumanDetectionOptions? options,
  }) async {
    await _ensureInitialized(options);
    return HumanDetectionPlatform.instance.detectHuman(imagePath);
  }

  /// Detect if image bytes contain a human.
  ///
  /// Useful for images from camera, network, or memory.
  ///
  /// Example:
  /// ```dart
  /// final bytes = await networkImage.readAsBytes();
  /// final result = await HumanDetection.detectFromBytes(bytes);
  /// ```
  static Future<HumanDetectionResult> detectFromBytes(
    Uint8List imageBytes, {
    HumanDetectionOptions? options,
  }) async {
    await _ensureInitialized(options);
    return HumanDetectionPlatform.instance.detectHumanFromBytes(imageBytes);
  }

  /// Configure detection options.
  ///
  /// Call this before detection to customize behavior.
  /// If not called, default options are used.
  ///
  /// Example:
  /// ```dart
  /// await HumanDetection.configure(HumanDetectionOptions(
  ///   confidenceThreshold: 0.7,
  ///   useGpuDelegate: true,
  /// ));
  /// ```
  static Future<void> configure(HumanDetectionOptions options) async {
    // If already initialized with different options, reinitialize
    if (_initialized && _currentOptions != options) {
      await dispose();
    }
    await _ensureInitialized(options);
  }

  /// Dispose resources and release memory.
  ///
  /// Call this when completely done with human detection.
  /// The next detect call will automatically re-initialize.
  static Future<void> dispose() async {
    if (!_initialized) return;
    await HumanDetectionPlatform.instance.dispose();
    _initialized = false;
    _initCompleter = null;
    _currentOptions = null;
  }

  /// Get the current platform version.
  static Future<String?> getPlatformVersion() {
    return HumanDetectionPlatform.instance.getPlatformVersion();
  }

  // Internal: Ensures model is initialized before detection
  static Future<void> _ensureInitialized([HumanDetectionOptions? options]) async {
    if (_initialized) return;

    // Handle concurrent initialization calls
    if (_initCompleter != null) {
      return _initCompleter!.future;
    }

    _initCompleter = Completer<void>();
    try {
      final opts = options ?? const HumanDetectionOptions();
      await HumanDetectionPlatform.instance.initialize(opts);
      _currentOptions = opts;
      _initialized = true;
      _initCompleter!.complete();
    } catch (e) {
      _initCompleter!.completeError(e);
      _initCompleter = null;
      rethrow;
    }
  }
}
