import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:human_detection/human_detection_method_channel.dart';
import 'package:human_detection/src/human_detection_options.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  MethodChannelHumanDetection platform = MethodChannelHumanDetection();
  const MethodChannel channel = MethodChannel('human_detection');

  setUp(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          switch (methodCall.method) {
            case 'getPlatformVersion':
              return '42';
            case 'initialize':
              return null;
            case 'detectHuman':
              return {
                'isHuman': true,
                'confidence': 0.9,
                'processingTimeMs': 50,
              };
            case 'detectHumanFromBytes':
              return {
                'isHuman': false,
                'confidence': 0.3,
                'processingTimeMs': 45,
              };
            case 'dispose':
              return null;
            default:
              return null;
          }
        });
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  test('getPlatformVersion', () async {
    expect(await platform.getPlatformVersion(), '42');
  });

  test('initialize completes without error', () async {
    await expectLater(
      platform.initialize(const HumanDetectionOptions()),
      completes,
    );
  });

  test('detectHuman returns correct result', () async {
    final result = await platform.detectHuman('/path/to/image.jpg');

    expect(result.isHuman, true);
    expect(result.confidence, 0.9);
    expect(result.processingTimeMs, 50);
  });

  test('detectHumanFromBytes returns correct result', () async {
    final result = await platform.detectHumanFromBytes(Uint8List(100));

    expect(result.isHuman, false);
    expect(result.confidence, 0.3);
    expect(result.processingTimeMs, 45);
  });

  test('dispose completes without error', () async {
    await expectLater(platform.dispose(), completes);
  });
}
