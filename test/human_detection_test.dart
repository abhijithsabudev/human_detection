import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:human_detection/human_detection.dart';
import 'package:human_detection/human_detection_platform_interface.dart';
import 'package:human_detection/human_detection_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockHumanDetectionPlatform
    with MockPlatformInterfaceMixin
    implements HumanDetectionPlatform {
  bool initialized = false;

  @override
  Future<String?> getPlatformVersion() => Future.value('42');

  @override
  Future<void> initialize(HumanDetectionOptions options) async {
    initialized = true;
  }

  @override
  Future<HumanDetectionResult> detectHuman(String imagePath) async {
    return const HumanDetectionResult(
      isHuman: true,
      confidence: 0.95,
      processingTimeMs: 50,
    );
  }

  @override
  Future<HumanDetectionResult> detectHumanFromBytes(
    Uint8List imageBytes,
  ) async {
    return const HumanDetectionResult(
      isHuman: false,
      confidence: 0.2,
      processingTimeMs: 45,
    );
  }

  @override
  Future<void> dispose() async {
    initialized = false;
  }
}

void main() {
  final HumanDetectionPlatform initialPlatform =
      HumanDetectionPlatform.instance;

  test('$MethodChannelHumanDetection is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelHumanDetection>());
  });

  group('HumanDetection static API', () {
    late MockHumanDetectionPlatform mockPlatform;

    setUp(() async {
      mockPlatform = MockHumanDetectionPlatform();
      HumanDetectionPlatform.instance = mockPlatform;
      // Reset state before each test
      await HumanDetection.dispose();
    });

    tearDown(() async {
      await HumanDetection.dispose();
    });

    test('detect auto-initializes and returns result', () async {
      expect(HumanDetection.isInitialized, false);

      final result = await HumanDetection.detect('/path/to/image.jpg');

      expect(HumanDetection.isInitialized, true);
      expect(result.isHuman, true);
      expect(result.confidence, 0.95);
      expect(result.processingTimeMs, 50);
    });

    test('detectFromBytes auto-initializes and returns result', () async {
      expect(HumanDetection.isInitialized, false);

      final result = await HumanDetection.detectFromBytes(Uint8List(100));

      expect(HumanDetection.isInitialized, true);
      expect(result.isHuman, false);
      expect(result.confidence, 0.2);
      expect(result.processingTimeMs, 45);
    });

    test('dispose resets initialization state', () async {
      await HumanDetection.detect('/path/to/image.jpg');
      expect(HumanDetection.isInitialized, true);

      await HumanDetection.dispose();
      expect(HumanDetection.isInitialized, false);
    });

    test('getPlatformVersion returns platform version', () async {
      final version = await HumanDetection.getPlatformVersion();
      expect(version, '42');
    });

    test('configure initializes with custom options', () async {
      await HumanDetection.configure(
        const HumanDetectionOptions(
          confidenceThreshold: 0.8,
          useGpuDelegate: false,
        ),
      );

      expect(HumanDetection.isInitialized, true);
      expect(mockPlatform.initialized, true);
    });
  });

  group('HumanDetectionResult', () {
    test('fromMap creates result correctly', () {
      final map = {
        'isHuman': true,
        'confidence': 0.85,
        'processingTimeMs': 100,
      };

      final result = HumanDetectionResult.fromMap(map);

      expect(result.isHuman, true);
      expect(result.confidence, 0.85);
      expect(result.processingTimeMs, 100);
    });

    test('toMap serializes correctly', () {
      const result = HumanDetectionResult(
        isHuman: true,
        confidence: 0.9,
        processingTimeMs: 75,
      );

      final map = result.toMap();

      expect(map['isHuman'], true);
      expect(map['confidence'], 0.9);
      expect(map['processingTimeMs'], 75);
    });

    test('toString provides readable output', () {
      const result = HumanDetectionResult(
        isHuman: true,
        confidence: 0.95,
        processingTimeMs: 50,
      );

      expect(result.toString(), contains('isHuman: true'));
      expect(result.toString(), contains('95.0%'));
    });
  });

  group('HumanDetectionOptions', () {
    test('default values are set correctly', () {
      const options = HumanDetectionOptions();

      expect(options.confidenceThreshold, 0.5);
      expect(options.useGpuDelegate, false);
      expect(options.numThreads, 4);
      expect(options.modelPath, null);
    });

    test('copyWith creates new instance with updated values', () {
      const options = HumanDetectionOptions();
      final updated = options.copyWith(confidenceThreshold: 0.7);

      expect(updated.confidenceThreshold, 0.7);
      expect(updated.useGpuDelegate, false); // unchanged
    });

    test('toMap serializes all properties', () {
      const options = HumanDetectionOptions(
        confidenceThreshold: 0.6,
        useGpuDelegate: false,
        numThreads: 2,
        modelPath: '/custom/model.tflite',
      );

      final map = options.toMap();

      expect(map['confidenceThreshold'], 0.6);
      expect(map['useGpuDelegate'], false);
      expect(map['numThreads'], 2);
      expect(map['modelPath'], '/custom/model.tflite');
    });
  });
}
