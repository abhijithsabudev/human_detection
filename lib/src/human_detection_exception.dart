/// Exception thrown when human detection operations fail.
class HumanDetectionException implements Exception {
  /// Creates a new [HumanDetectionException].
  const HumanDetectionException(this.message, {this.code, this.details});

  /// Creates a [HumanDetectionException] from a platform exception.
  factory HumanDetectionException.fromPlatformException(dynamic e) {
    if (e is HumanDetectionException) return e;
    return HumanDetectionException(e.toString(), details: e);
  }

  /// Error message describing what went wrong.
  final String message;

  /// Error code for programmatic handling.
  final String? code;

  /// Additional details about the error.
  final dynamic details;

  @override
  String toString() {
    final buffer = StringBuffer('HumanDetectionException: $message');
    if (code != null) {
      buffer.write(' (code: $code)');
    }
    return buffer.toString();
  }
}

/// Exception codes for common error scenarios.
class HumanDetectionErrorCode {
  HumanDetectionErrorCode._();

  /// Model initialization failed.
  static const String initializationFailed = 'INITIALIZATION_FAILED';

  /// Model file not found.
  static const String modelNotFound = 'MODEL_NOT_FOUND';

  /// Image file not found.
  static const String imageNotFound = 'IMAGE_NOT_FOUND';

  /// Invalid image format.
  static const String invalidImageFormat = 'INVALID_IMAGE_FORMAT';

  /// Inference failed.
  static const String inferenceFailed = 'INFERENCE_FAILED';

  /// Not initialized.
  static const String notInitialized = 'NOT_INITIALIZED';

  /// Platform not supported.
  static const String platformNotSupported = 'PLATFORM_NOT_SUPPORTED';
}
