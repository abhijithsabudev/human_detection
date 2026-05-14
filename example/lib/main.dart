import 'dart:io';

import 'package:flutter/material.dart';
import 'package:human_detection/human_detection.dart';
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Human Detection Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const HumanDetectionDemo(),
    );
  }
}

class HumanDetectionDemo extends StatefulWidget {
  const HumanDetectionDemo({super.key});

  @override
  State<HumanDetectionDemo> createState() => _HumanDetectionDemoState();
}

class _HumanDetectionDemoState extends State<HumanDetectionDemo> {
  final ImagePicker _imagePicker = ImagePicker();

  bool _isLoading = false;
  File? _selectedImage;
  HumanDetectionResult? _detectionResult;
  String? _errorMessage;

  Future<void> _pickImage(ImageSource source) async {
    try {
      final pickedFile = await _imagePicker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
      );

      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
          _detectionResult = null;
          _errorMessage = null;
        });
        // Auto-detect after picking image
        await _detectHuman();
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to pick image: $e';
      });
    }
  }

  Future<void> _detectHuman() async {
    if (_selectedImage == null) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _detectionResult = null;
    });

    try {
      // Simple one-liner! No initialization required.
      final result = await HumanDetection.detect(_selectedImage!.path);

      if (!mounted) return;

      setState(() {
        _detectionResult = result;
        _isLoading = false;
      });
    } on HumanDetectionException catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Detection error: ${e.message}';
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Error: $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Human Detection'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image selection buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isLoading
                        ? null
                        : () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isLoading
                        ? null
                        : () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),

            // Selected image display
            if (_selectedImage != null) ...[
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.file(
                  _selectedImage!,
                  height: 300,
                  width: double.infinity,
                  fit: BoxFit.cover,
                ),
              ),
              const SizedBox(height: 16),
            ],

            // Loading indicator
            if (_isLoading)
              const Center(
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 8),
                    Text('Detecting...'),
                  ],
                ),
              ),

            // Error message
            if (_errorMessage != null)
              Card(
                color: Colors.red.shade100,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      Icon(Icons.error, color: Colors.red.shade700),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _errorMessage!,
                          style: TextStyle(color: Colors.red.shade700),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

            // Detection result
            if (_detectionResult != null) _buildResultCard(_detectionResult!),

            const SizedBox(height: 24),

            // Info card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'How it works',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'This demo uses the human_detection package which '
                      'automatically loads a pre-trained ML model to detect '
                      'humans in images. Just select an image and the detection '
                      'happens automatically!',
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard(HumanDetectionResult result) {
    final isHuman = result.isHuman;
    final confidence = result.confidence;
    final processingTime = result.processingTimeMs;

    return Card(
      color: isHuman ? Colors.green.shade50 : Colors.orange.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  isHuman ? Icons.person : Icons.person_off,
                  size: 48,
                  color: isHuman ? Colors.green : Colors.orange,
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        isHuman ? 'Human Detected!' : 'No Human Detected',
                        style: Theme.of(context).textTheme.headlineSmall
                            ?.copyWith(
                              color: isHuman
                                  ? Colors.green.shade700
                                  : Colors.orange.shade700,
                              fontWeight: FontWeight.bold,
                            ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Confidence: ${(confidence * 100).toStringAsFixed(1)}%',
                        style: Theme.of(context).textTheme.bodyLarge,
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            // Confidence bar
            LinearProgressIndicator(
              value: confidence,
              backgroundColor: Colors.grey.shade300,
              valueColor: AlwaysStoppedAnimation<Color>(
                isHuman ? Colors.green : Colors.orange,
              ),
            ),
            if (processingTime != null) ...[
              const SizedBox(height: 8),
              Text(
                'Processing time: ${processingTime}ms',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
            if (result.boundingBox != null) ...[
              const SizedBox(height: 8),
              Text(
                'Bounding box: ${result.boundingBox}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ],
        ),
      ),
    );
  }
}
