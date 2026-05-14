#!/usr/bin/env python3
"""
Human Detection Model Training with Real Data

This script trains a human detection model using real datasets:
- Positive samples: Images containing humans (from COCO People, LFW, etc.)
- Negative samples: Images without humans (from Places365, ImageNet subsets, etc.)

Prerequisites:
    pip install -r requirements.txt

Usage:
    python train_with_real_data.py --data_dir /path/to/data

Data Directory Structure:
    data/
    ├── train/
    │   ├── human/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── non_human/
    │       ├── image1.jpg
    │       └── ...
    └── validation/
        ├── human/
        └── non_human/
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

def create_data_generators(data_dir):
    """
    Create data generators for training and validation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['non_human', 'human']
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['non_human', 'human']
    )
    
    return train_generator, val_generator

def create_model():
    """
    Create a human detection model using MobileNetV2 transfer learning.
    """
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model, base_model

def fine_tune_model(model, base_model):
    """
    Fine-tune the model by unfreezing top layers.
    """
    base_model.trainable = True
    
    # Freeze all layers except the last 50
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def plot_training_history(history, output_path):
    """
    Plot training metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training history plot saved to: {output_path}")

def convert_to_tflite(model, output_dir):
    """
    Convert model to TFLite with different optimization levels.
    """
    # Standard conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    standard_path = os.path.join(output_dir, 'human_detection_model.tflite')
    with open(standard_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Standard TFLite model: {standard_path} ({len(tflite_model)/1024/1024:.2f} MB)")
    
    # Float16 quantization (good balance of size and accuracy)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter.convert()
    
    fp16_path = os.path.join(output_dir, 'human_detection_model_fp16.tflite')
    with open(fp16_path, 'wb') as f:
        f.write(tflite_fp16)
    print(f"FP16 TFLite model: {fp16_path} ({len(tflite_fp16)/1024/1024:.2f} MB)")
    
    # Int8 quantization (smallest size, requires representative dataset)
    # Note: For production, you should provide a representative dataset
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_int8 = converter.convert()
    
    int8_path = os.path.join(output_dir, 'human_detection_model_int8.tflite')
    with open(int8_path, 'wb') as f:
        f.write(tflite_int8)
    print(f"Int8 TFLite model: {int8_path} ({len(tflite_int8)/1024/1024:.2f} MB)")
    
    return standard_path

def main():
    parser = argparse.ArgumentParser(description='Train human detection model')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Path to output directory')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Human Detection Model Training (Real Data)")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"\nError: Data directory '{args.data_dir}' does not exist.")
        print("\nPlease create the following structure:")
        print("  data/")
        print("  ├── train/")
        print("  │   ├── human/     (images with humans)")
        print("  │   └── non_human/ (images without humans)")
        print("  └── validation/")
        print("      ├── human/")
        print("      └── non_human/")
        print("\nYou can download datasets from:")
        print("  - COCO: https://cocodataset.org/")
        print("  - LFW: http://vis-www.cs.umass.edu/lfw/")
        print("  - Places365: http://places2.csail.mit.edu/")
        return
    
    # Create data generators
    train_gen, val_gen = create_data_generators(args.data_dir)
    
    print(f"\nTraining samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {train_gen.class_indices}")
    
    # Create and train model
    model, base_model = create_model()
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, 'best_model.h5'),
            save_best_only=True
        )
    ]
    
    # Initial training
    print("\n[Phase 1] Training classification head...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs // 2,
        callbacks=callbacks
    )
    
    # Fine-tuning
    print("\n[Phase 2] Fine-tuning...")
    model = fine_tune_model(model, base_model)
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs // 2,
        callbacks=callbacks
    )
    
    # Combine histories
    history = history1
    for key in history.history:
        history.history[key].extend(history2.history[key])
    
    # Plot training history
    plot_training_history(history, os.path.join(args.output_dir, 'training_history.png'))
    
    # Save final model
    model.save(os.path.join(args.output_dir, 'human_detection_model.h5'))
    
    # Convert to TFLite
    convert_to_tflite(model, args.output_dir)
    
    # Create labels file
    with open(os.path.join(args.output_dir, 'labels.txt'), 'w') as f:
        f.write('non_human\nhuman')
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
