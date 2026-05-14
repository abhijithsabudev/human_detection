/// Result of human detection analysis.
///
/// Contains information about whether a human was detected,
/// the confidence level, and processing metrics.
class HumanDetectionResult {
  /// Creates a new [HumanDetectionResult].
  const HumanDetectionResult({
    required this.isHuman,
    required this.confidence,
    this.processingTimeMs,
    this.boundingBox,
  });

  /// Creates a [HumanDetectionResult] from a map.
  ///
  /// Used for deserialization from platform channels.
  factory HumanDetectionResult.fromMap(Map<dynamic, dynamic> map) {
    return HumanDetectionResult(
      isHuman: map['isHuman'] as bool? ?? false,
      confidence: (map['confidence'] as num?)?.toDouble() ?? 0.0,
      processingTimeMs: map['processingTimeMs'] as int?,
      boundingBox: map['boundingBox'] != null
          ? BoundingBox.fromMap(map['boundingBox'] as Map<dynamic, dynamic>)
          : null,
    );
  }

  /// Whether a human was detected in the image.
  ///
  /// This is `true` if the confidence exceeds the threshold
  /// specified in [HumanDetectionOptions].
  final bool isHuman;

  /// Confidence score of the detection (0.0 to 1.0).
  ///
  /// - 1.0 means very confident that a human is present
  /// - 0.0 means very confident that no human is present
  /// - Values around 0.5 indicate uncertainty
  final double confidence;

  /// Time taken to process the image in milliseconds.
  ///
  /// This can be used for performance monitoring.
  final int? processingTimeMs;

  /// Bounding box of the detected human (if available).
  ///
  /// This is only available when using models that support localization.
  final BoundingBox? boundingBox;

  /// Converts this result to a map.
  Map<String, dynamic> toMap() {
    return {
      'isHuman': isHuman,
      'confidence': confidence,
      'processingTimeMs': processingTimeMs,
      'boundingBox': boundingBox?.toMap(),
    };
  }

  @override
  String toString() {
    return 'HumanDetectionResult('
        'isHuman: $isHuman, '
        'confidence: ${(confidence * 100).toStringAsFixed(1)}%, '
        'processingTimeMs: ${processingTimeMs}ms'
        ')';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is HumanDetectionResult &&
        other.isHuman == isHuman &&
        other.confidence == confidence &&
        other.processingTimeMs == processingTimeMs;
  }

  @override
  int get hashCode => Object.hash(isHuman, confidence, processingTimeMs);
}

/// Bounding box representing the location of a detected human.
class BoundingBox {
  /// Creates a new [BoundingBox].
  const BoundingBox({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  /// Creates a [BoundingBox] from a map.
  factory BoundingBox.fromMap(Map<dynamic, dynamic> map) {
    return BoundingBox(
      left: (map['left'] as num).toDouble(),
      top: (map['top'] as num).toDouble(),
      right: (map['right'] as num).toDouble(),
      bottom: (map['bottom'] as num).toDouble(),
    );
  }

  /// Left edge of the bounding box (normalized 0.0 to 1.0).
  final double left;

  /// Top edge of the bounding box (normalized 0.0 to 1.0).
  final double top;

  /// Right edge of the bounding box (normalized 0.0 to 1.0).
  final double right;

  /// Bottom edge of the bounding box (normalized 0.0 to 1.0).
  final double bottom;

  /// Width of the bounding box.
  double get width => right - left;

  /// Height of the bounding box.
  double get height => bottom - top;

  /// Converts this bounding box to a map.
  Map<String, dynamic> toMap() {
    return {'left': left, 'top': top, 'right': right, 'bottom': bottom};
  }

  @override
  String toString() {
    return 'BoundingBox(left: $left, top: $top, right: $right, bottom: $bottom)';
  }
}
