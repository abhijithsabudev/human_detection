import 'dart:typed_data';

import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'human_detection_method_channel.dart';
import 'src/human_detection_options.dart';
import 'src/human_detection_result.dart';

abstract class HumanDetectionPlatform extends PlatformInterface {
  /// Constructs a HumanDetectionPlatform.
  HumanDetectionPlatform() : super(token: _token);

  static final Object _token = Object();

  static HumanDetectionPlatform _instance = MethodChannelHumanDetection();

  /// The default instance of [HumanDetectionPlatform] to use.
  ///
  /// Defaults to [MethodChannelHumanDetection].
  static HumanDetectionPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [HumanDetectionPlatform] when
  /// they register themselves.
  static set instance(HumanDetectionPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  /// Initialize the human detection model.
  Future<void> initialize(HumanDetectionOptions options) {
    throw UnimplementedError('initialize() has not been implemented.');
  }

  /// Detect human in image at the given path.
  Future<HumanDetectionResult> detectHuman(String imagePath) {
    throw UnimplementedError('detectHuman() has not been implemented.');
  }

  /// Detect human from image bytes.
  Future<HumanDetectionResult> detectHumanFromBytes(Uint8List imageBytes) {
    throw UnimplementedError(
      'detectHumanFromBytes() has not been implemented.',
    );
  }

  /// Dispose of resources.
  Future<void> dispose() {
    throw UnimplementedError('dispose() has not been implemented.');
  }

  /// Get platform version.
  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
