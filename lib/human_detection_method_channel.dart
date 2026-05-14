import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'human_detection_platform_interface.dart';
import 'src/human_detection_exception.dart';
import 'src/human_detection_options.dart';
import 'src/human_detection_result.dart';

/// An implementation of [HumanDetectionPlatform] that uses method channels.
class MethodChannelHumanDetection extends HumanDetectionPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('human_detection');

  @override
  Future<void> initialize(HumanDetectionOptions options) async {
    try {
      await methodChannel.invokeMethod<void>('initialize', options.toMap());
    } on PlatformException catch (e) {
      throw HumanDetectionException(
        e.message ?? 'Failed to initialize human detection',
        code: HumanDetectionErrorCode.initializationFailed,
        details: e.details,
      );
    }
  }

  @override
  Future<HumanDetectionResult> detectHuman(String imagePath) async {
    try {
      final result = await methodChannel.invokeMapMethod<dynamic, dynamic>(
        'detectHuman',
        {'imagePath': imagePath},
      );
      if (result == null) {
        throw const HumanDetectionException(
          'Detection returned null result',
          code: HumanDetectionErrorCode.inferenceFailed,
        );
      }
      return HumanDetectionResult.fromMap(result);
    } on PlatformException catch (e) {
      throw HumanDetectionException(
        e.message ?? 'Failed to detect human',
        code: e.code,
        details: e.details,
      );
    }
  }

  @override
  Future<HumanDetectionResult> detectHumanFromBytes(
    Uint8List imageBytes,
  ) async {
    try {
      final result = await methodChannel.invokeMapMethod<dynamic, dynamic>(
        'detectHumanFromBytes',
        {'imageBytes': imageBytes},
      );
      if (result == null) {
        throw const HumanDetectionException(
          'Detection returned null result',
          code: HumanDetectionErrorCode.inferenceFailed,
        );
      }
      return HumanDetectionResult.fromMap(result);
    } on PlatformException catch (e) {
      throw HumanDetectionException(
        e.message ?? 'Failed to detect human from bytes',
        code: e.code,
        details: e.details,
      );
    }
  }

  @override
  Future<void> dispose() async {
    try {
      await methodChannel.invokeMethod<void>('dispose');
    } on PlatformException catch (e) {
      throw HumanDetectionException(
        e.message ?? 'Failed to dispose',
        details: e.details,
      );
    }
  }

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>(
      'getPlatformVersion',
    );
    return version;
  }
}
