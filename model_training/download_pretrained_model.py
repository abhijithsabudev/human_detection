#!/usr/bin/env python3
"""
Download pre-trained person detection model from TensorFlow Hub.

Uses EfficientDet-Lite or SSD MobileNet trained on COCO dataset.
The "person" class (index 0 in COCO) is used for human detection.
"""

import os
import urllib.request
import zipfile

# Pre-trained model URLs (TensorFlow Lite format)
MODELS = {
    # SSD MobileNet V1 - Fast, good accuracy
    'ssd_mobilenet_v1': {
        'url': 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip',
        'filename': 'detect.tflite',
        'description': 'SSD MobileNet V1 (COCO) - Fast, quantized'
    },
    # EfficientDet Lite0 - Best balance of speed/accuracy
    'efficientdet_lite0': {
        'url': 'https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/default/1?lite-format=tflite',
        'filename': 'efficientdet_lite0.tflite',
        'description': 'EfficientDet Lite0 - Balanced speed/accuracy',
        'direct': True
    },
    # MobileNet V2 SSD - Good for mobile
    'ssd_mobilenet_v2': {
        'url': 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite',
        'filename': 'ssd_mobilenet_v2.tflite',
        'description': 'SSD MobileNet V2 with metadata',
        'direct': True
    }
}

def download_file(url, output_path, direct=False):
    """Download a file from URL."""
    print(f"Downloading from: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def download_model(model_name='ssd_mobilenet_v2'):
    """Download the specified pre-trained model."""
    
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {list(MODELS.keys())}")
        return None
    
    model_info = MODELS[model_name]
    print(f"\nDownloading: {model_info['description']}")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(script_dir, 'output')
    
    android_assets = os.path.join(parent_dir, 'android', 'src', 'main', 'assets')
    ios_assets = os.path.join(parent_dir, 'ios', 'Assets')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(android_assets, exist_ok=True)
    os.makedirs(ios_assets, exist_ok=True)
    
    # Download
    if model_info.get('direct', False):
        # Direct TFLite download
        output_path = os.path.join(output_dir, 'human_detection_model.tflite')
        if download_file(model_info['url'], output_path):
            # Copy to asset directories
            import shutil
            android_path = os.path.join(android_assets, 'human_detection_model.tflite')
            ios_path = os.path.join(ios_assets, 'human_detection_model.tflite')
            
            shutil.copy(output_path, android_path)
            shutil.copy(output_path, ios_path)
            
            print(f"\n✓ Model saved to:")
            print(f"  - {android_path}")
            print(f"  - {ios_path}")
            print(f"  - {output_path}")
            
            return output_path
    else:
        # ZIP download
        zip_path = os.path.join(output_dir, 'model.zip')
        if download_file(model_info['url'], zip_path):
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path)
            
            # Find and rename the tflite file
            source_path = os.path.join(output_dir, model_info['filename'])
            output_path = os.path.join(output_dir, 'human_detection_model.tflite')
            
            if os.path.exists(source_path):
                os.rename(source_path, output_path)
                
                # Copy to asset directories
                import shutil
                android_path = os.path.join(android_assets, 'human_detection_model.tflite')
                ios_path = os.path.join(ios_assets, 'human_detection_model.tflite')
                
                shutil.copy(output_path, android_path)
                shutil.copy(output_path, ios_path)
                
                print(f"\n✓ Model saved to:")
                print(f"  - {android_path}")
                print(f"  - {ios_path}")
                print(f"  - {output_path}")
                
                return output_path
    
    return None

def verify_model(model_path):
    """Verify the downloaded model."""
    try:
        import tensorflow as tf
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\nModel verification:")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        print(f"  Outputs: {len(output_details)}")
        
        for i, out in enumerate(output_details):
            print(f"    Output {i}: {out['shape']} ({out['dtype']})")
        
        # Get model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Model size: {model_size:.2f} MB")
        
        return True
    except Exception as e:
        print(f"Verification error: {e}")
        return False

def main():
    print("=" * 60)
    print("Pre-trained Human Detection Model Downloader")
    print("=" * 60)
    print("\nThis downloads a pre-trained object detection model from")
    print("TensorFlow Hub that can detect people (COCO class 0).")
    
    # Download the model
    model_path = download_model('ssd_mobilenet_v2')
    
    if model_path:
        verify_model(model_path)
        
        print("\n" + "=" * 60)
        print("✓ Download complete!")
        print("=" * 60)
        print("\nThe model can detect 80 COCO classes including 'person' (class 0).")
        print("The native code filters for person detections only.")
    else:
        print("\n✗ Download failed. Try a different model or check your connection.")

if __name__ == '__main__':
    main()
