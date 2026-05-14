/// Configuration options for human detection.
///
/// Use these options to customize the behavior of the detector.
class HumanDetectionOptions {
  /// Creates new [HumanDetectionOptions].
  const HumanDetectionOptions({
    this.confidenceThreshold = 0.5,
    this.useGpuDelegate = true,
    this.numThreads = 4,
    this.modelPath,
  });

  /// Minimum confidence threshold to consider a detection as human.
  ///
  /// Values should be between 0.0 and 1.0.
  /// - Lower values (e.g., 0.3) will be more sensitive but may have more false positives
  /// - Higher values (e.g., 0.7) will be more strict but may miss some humans
  ///
  /// Default is 0.5.
  final double confidenceThreshold;

  /// Whether to use GPU acceleration when available.
  ///
  /// GPU acceleration can significantly improve performance on supported devices.
  /// Default is `true`.
  final bool useGpuDelegate;

  /// Number of threads to use for inference.
  ///
  /// More threads can improve performance on multi-core devices.
  /// Default is 4.
  final int numThreads;

  /// Custom path to a TFLite model file.
  ///
  /// If not specified, the bundled model will be used.
  /// This allows using custom-trained models.
  final String? modelPath;

  /// Converts this options object to a map.
  Map<String, dynamic> toMap() {
    return {
      'confidenceThreshold': confidenceThreshold,
      'useGpuDelegate': useGpuDelegate,
      'numThreads': numThreads,
      'modelPath': modelPath,
    };
  }

  /// Creates a copy of this options with the given fields replaced.
  HumanDetectionOptions copyWith({
    double? confidenceThreshold,
    bool? useGpuDelegate,
    int? numThreads,
    String? modelPath,
  }) {
    return HumanDetectionOptions(
      confidenceThreshold: confidenceThreshold ?? this.confidenceThreshold,
      useGpuDelegate: useGpuDelegate ?? this.useGpuDelegate,
      numThreads: numThreads ?? this.numThreads,
      modelPath: modelPath ?? this.modelPath,
    );
  }

  @override
  String toString() {
    return 'HumanDetectionOptions('
        'confidenceThreshold: $confidenceThreshold, '
        'useGpuDelegate: $useGpuDelegate, '
        'numThreads: $numThreads, '
        'modelPath: $modelPath'
        ')';
  }
}
