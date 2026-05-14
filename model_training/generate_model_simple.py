#!/usr/bin/env python3
"""
Generate a minimal TFLite model without training.
This avoids CPU compatibility issues by skipping the training step.
"""

import os
import numpy as np

# Disable TensorFlow optimizations that cause issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def create_simple_model():
    """Create a simple model using tf.function for conversion."""
    
    class SimpleHumanDetector(tf.Module):
        def __init__(self):
            super().__init__()
            # Initialize weights with random values
            self.conv1_w = tf.Variable(tf.random.normal([3, 3, 3, 16], stddev=0.1), name='conv1_w')
            self.conv1_b = tf.Variable(tf.zeros([16]), name='conv1_b')
            
            self.conv2_w = tf.Variable(tf.random.normal([3, 3, 16, 32], stddev=0.1), name='conv2_w')
            self.conv2_b = tf.Variable(tf.zeros([32]), name='conv2_b')
            
            self.conv3_w = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.1), name='conv3_w')
            self.conv3_b = tf.Variable(tf.zeros([64]), name='conv3_b')
            
            self.dense1_w = tf.Variable(tf.random.normal([64, 64], stddev=0.1), name='dense1_w')
            self.dense1_b = tf.Variable(tf.zeros([64]), name='dense1_b')
            
            self.dense2_w = tf.Variable(tf.random.normal([64, 1], stddev=0.1), name='dense2_w')
            self.dense2_b = tf.Variable(tf.zeros([1]), name='dense2_b')
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)])
        def __call__(self, x):
            # Conv block 1
            x = tf.nn.conv2d(x, self.conv1_w, strides=[1, 2, 2, 1], padding='SAME')
            x = tf.nn.bias_add(x, self.conv1_b)
            x = tf.nn.relu(x)
            
            # Conv block 2
            x = tf.nn.conv2d(x, self.conv2_w, strides=[1, 2, 2, 1], padding='SAME')
            x = tf.nn.bias_add(x, self.conv2_b)
            x = tf.nn.relu(x)
            
            # Conv block 3
            x = tf.nn.conv2d(x, self.conv3_w, strides=[1, 2, 2, 1], padding='SAME')
            x = tf.nn.bias_add(x, self.conv3_b)
            x = tf.nn.relu(x)
            
            # Global average pooling
            x = tf.reduce_mean(x, axis=[1, 2])
            
            # Dense layers
            x = tf.matmul(x, self.dense1_w) + self.dense1_b
            x = tf.nn.relu(x)
            
            x = tf.matmul(x, self.dense2_w) + self.dense2_b
            x = tf.nn.sigmoid(x)
            
            return x
    
    return SimpleHumanDetector()

def main():
    print("=" * 60)
    print("Generating Minimal TFLite Model (No Training)")
    print("=" * 60)
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    android_assets = os.path.join(parent_dir, 'android', 'src', 'main', 'assets')
    ios_assets = os.path.join(parent_dir, 'ios', 'Assets')
    output_dir = os.path.join(script_dir, 'output')
    
    os.makedirs(android_assets, exist_ok=True)
    os.makedirs(ios_assets, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nCreating model...")
    model = create_simple_model()
    
    # Test the model
    print("Testing model...")
    test_input = tf.random.normal([1, 224, 224, 3])
    output = model(test_input)
    print(f"  Test output shape: {output.shape}")
    print(f"  Test output value: {output.numpy()[0][0]:.4f}")
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    concrete_func = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save to all locations
    android_path = os.path.join(android_assets, 'human_detection_model.tflite')
    ios_path = os.path.join(ios_assets, 'human_detection_model.tflite')
    output_path = os.path.join(output_dir, 'human_detection_model.tflite')
    
    for path in [android_path, ios_path, output_path]:
        with open(path, 'wb') as f:
            f.write(tflite_model)
        print(f"  Saved: {path}")
    
    print(f"\nModel size: {len(tflite_model) / 1024:.2f} KB")
    
    # Verify model
    print("\nVerifying model...")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    # Test inference
    test_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_data)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    print(f"  Test inference result: {result[0][0]:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Model generation complete!")
    print("=" * 60)
    print("\nNOTE: This is a minimal model with random weights.")
    print("For production, train with real data using train_with_real_data.py")

if __name__ == '__main__':
    main()
