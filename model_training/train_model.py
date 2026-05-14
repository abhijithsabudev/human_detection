#!/usr/bin/env python3
"""
Human Detection Model Training Script

This script trains a binary classifier to detect if an image contains a human or not.
Uses MobileNetV2 with transfer learning for efficient mobile deployment.

Usage:
    python train_model.py

Output:
    - human_detection_model.tflite (TensorFlow Lite model for mobile)
    - human_detection_model.h5 (Full Keras model for reference)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
MODEL_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_dataset():
    """
    Create training and validation datasets.
    Uses COCO dataset (humans) and other datasets (non-humans).
    """
    print("Loading datasets...")
    
    # Load COCO People dataset for human images
    # We'll use a subset of images that contain people
    try:
        # Try to load from TensorFlow Datasets
        coco_dataset = tfds.load('coco/2017', split='train', as_supervised=False)
        
        # Filter for images containing people (category_id 1 is person in COCO)
        def has_person(example):
            labels = example['objects']['label']
            return tf.reduce_any(tf.equal(labels, 0))  # 0 is person in TFDS COCO
        
        human_dataset = coco_dataset.filter(has_person)
        
    except Exception as e:
        print(f"Could not load COCO dataset: {e}")
        print("Using synthetic data for demonstration...")
        return create_synthetic_dataset()
    
    return prepare_dataset(human_dataset)

def create_synthetic_dataset():
    """
    Create a synthetic dataset for demonstration.
    In production, you should use real datasets like COCO, ImageNet, etc.
    """
    print("Creating synthetic dataset for demonstration...")
    
    # Generate synthetic data
    np.random.seed(42)
    num_samples = 1000
    
    # Create random images (in production, use real images)
    X = np.random.rand(num_samples, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    y = np.random.randint(0, 2, num_samples).astype(np.float32)
    
    # Split into train and validation
    split_idx = int(0.8 * num_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

def prepare_dataset(dataset):
    """
    Prepare dataset for training.
    """
    def preprocess(example):
        image = example['image']
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32) / 255.0
        
        # Check if person is in the image
        labels = example['objects']['label']
        has_person = tf.cast(tf.reduce_any(tf.equal(labels, 0)), tf.float32)
        
        return image, has_person
    
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Split into train and validation
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    train_size = int(0.8 * dataset_size)
    
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

def create_model():
    """
    Create a human detection model using MobileNetV2 transfer learning.
    MobileNetV2 is optimized for mobile/edge devices.
    """
    print("Creating model with MobileNetV2 backbone...")
    
    # Load pre-trained MobileNetV2 (without top layers)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create the model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Data augmentation layers
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Preprocessing for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model, base_model

def fine_tune_model(model, base_model, train_dataset, val_dataset):
    """
    Fine-tune the model by unfreezing some layers of the base model.
    """
    print("Fine-tuning model...")
    
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Continue training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10
    )
    
    return history

def convert_to_tflite(model, output_path):
    """
    Convert Keras model to TensorFlow Lite format for mobile deployment.
    """
    print("Converting to TensorFlow Lite...")
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization for mobile
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    return tflite_model

def create_labels_file(output_path):
    """
    Create labels file for the model.
    """
    labels = ["non_human", "human"]
    with open(output_path, 'w') as f:
        f.write('\n'.join(labels))
    print(f"Labels file saved to: {output_path}")

def main():
    print("=" * 60)
    print("Human Detection Model Training")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(MODEL_OUTPUT_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets
    train_dataset, val_dataset = create_synthetic_dataset()
    
    # Create model
    model, base_model = create_model()
    model.summary()
    
    # Train model
    print("\nTraining model...")
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
    ]
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Fine-tune
    fine_tune_model(model, base_model, train_dataset, val_dataset)
    
    # Save Keras model
    keras_path = os.path.join(output_dir, 'human_detection_model.h5')
    model.save(keras_path)
    print(f"Keras model saved to: {keras_path}")
    
    # Convert to TFLite
    tflite_path = os.path.join(output_dir, 'human_detection_model.tflite')
    convert_to_tflite(model, tflite_path)
    
    # Create labels file
    labels_path = os.path.join(output_dir, 'labels.txt')
    create_labels_file(labels_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Keras model: {keras_path}")
    print(f"  - TFLite model: {tflite_path}")
    print(f"  - Labels: {labels_path}")
    print(f"\nCopy the TFLite model to:")
    print(f"  - Android: android/src/main/assets/")
    print(f"  - iOS: ios/Assets/")

if __name__ == '__main__':
    main()
