import 'dart:io';
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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
  final HumanDetection _humanDetection = HumanDetection();
  final ImagePicker _imagePicker = ImagePicker();

  String _platformVersion = 'Unknown';
  bool _isInitialized = false;
  bool _isLoading = false;
  File? _selectedImage;
  HumanDetectionResult? _detectionResult;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initializePlugin();
  }

  @override
  void dispose() {
    _humanDetection.dispose();
    super.dispose();
  }

  Future<void> _initializePlugin() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      // Get platform version
      final platformVersion =
          await _humanDetection.getPlatformVersion() ??
          'Unknown platform version';

      // Initialize the human detection model
      await _humanDetection.initialize(
        const HumanDetectionOptions(
          confidenceThreshold: 0.5,
          useGpuDelegate: true,
          numThreads: 4,
        ),
      );

      if (!mounted) return;

      setState(() {
        _platformVersion = platformVersion;
        _isInitialized = true;
        _isLoading = false;
      });
    } on PlatformException catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Platform error: ${e.message}';
        _isLoading = false;
      });
    } on HumanDetectionException catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Initialization error: ${e.message}';
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Unknown error: $e';
        _isLoading = false;
      });
    }
  }

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
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to pick image: $e';
      });
    }
  }

  Future<void> _detectHuman() async {
    if (_selectedImage == null || !_isInitialized) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _detectionResult = null;
    });

    try {
      final result = await _humanDetection.detectHuman(_selectedImage!.path);

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
        _errorMessage = 'Unknown error: $e';
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
            // Platform info card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Platform: $_platformVersion',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Icon(
                          _isInitialized ? Icons.check_circle : Icons.pending,
                          color: _isInitialized ? Colors.green : Colors.orange,
                          size: 20,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          _isInitialized
                              ? 'Model Ready'
                              : 'Model Not Initialized',
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

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

            // Detect button
            if (_selectedImage != null)
              ElevatedButton.icon(
                onPressed: (_isLoading || !_isInitialized)
                    ? null
                    : _detectHuman,
                icon: _isLoading
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.search),
                label: Text(_isLoading ? 'Detecting...' : 'Detect Human'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
              ),

            const SizedBox(height: 16),

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
              minHeight: 8,
              borderRadius: BorderRadius.circular(4),
            ),
            const SizedBox(height: 16),
            // Processing time
            if (processingTime != null)
              Text(
                'Processing time: ${processingTime}ms',
                style: Theme.of(
                  context,
                ).textTheme.bodySmall?.copyWith(color: Colors.grey.shade600),
              ),
          ],
        ),
      ),
    );
  }
}
