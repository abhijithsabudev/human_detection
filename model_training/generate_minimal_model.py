#!/usr/bin/env python3
"""
Generate a minimal TFLite model for testing/bundling.

This script creates a small working TFLite model that can be used
for testing the human detection package. For production use,
train a proper model using train_with_real_data.py.

Usage:
    python generate_minimal_model.py
"""

import os
import numpy as np
import tensorflow as tf

# Use keras 3 API (TensorFlow 2.16+)
try:
    import keras
    from keras import layers
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers

def create_minimal_model():
    """Create a minimal model for testing."""
    
    # Input shape: 224x224x3 (standard for MobileNet)
    inputs = keras.Input(shape=(224, 224, 3), name='input')
    
    # Simple CNN architecture
    x = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def convert_to_tflite(model, output_path):
    """Convert Keras model to TFLite format."""
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model saved to: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_model

def verify_model(model_path):
    """Verify the TFLite model works correctly."""
    
    # Load and test the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nModel verification:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")
    
    # Run a test inference
    test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"  Test output: {output[0][0]:.4f}")
    print("  ✓ Model verification passed!")

def main():
    print("=" * 60)
    print("Generating Minimal TFLite Model")
    print("=" * 60)
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Create output directories
    android_assets = os.path.join(parent_dir, 'android', 'src', 'main', 'assets')
    ios_assets = os.path.join(parent_dir, 'ios', 'Assets')
    
    os.makedirs(android_assets, exist_ok=True)
    os.makedirs(ios_assets, exist_ok=True)
    
    # Create model
    print("\nCreating minimal model...")
    model = create_minimal_model()
    model.summary()
    
    # Train on random data (just to initialize weights properly)
    print("\nInitializing model weights...")
    X_dummy = np.random.rand(100, 224, 224, 3).astype(np.float32)
    y_dummy = np.random.randint(0, 2, 100).astype(np.float32)
    model.fit(X_dummy, y_dummy, epochs=1, verbose=0)
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    android_path = os.path.join(android_assets, 'human_detection_model.tflite')
    convert_to_tflite(model, android_path)
    
    # Copy to iOS assets
    ios_path = os.path.join(ios_assets, 'human_detection_model.tflite')
    import shutil
    shutil.copy(android_path, ios_path)
    print(f"Model copied to: {ios_path}")
    
    # Also save to output directory for reference
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'human_detection_model.tflite')
    shutil.copy(android_path, output_path)
    
    # Verify model
    verify_model(android_path)
    
    print("\n" + "=" * 60)
    print("Model Generation Complete!")
    print("=" * 60)
    print(f"\nModel files created:")
    print(f"  - {android_path}")
    print(f"  - {ios_path}")
    print(f"  - {output_path}")
    print("\nNOTE: This is a minimal model for testing.")
    print("For production, train a proper model using train_with_real_data.py")

if __name__ == '__main__':
    main()
