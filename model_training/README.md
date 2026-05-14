# Human Detection Model Training

This directory contains scripts for training the human detection machine learning model.

## Overview

The model uses **MobileNetV2** as the backbone with transfer learning, optimized for mobile deployment. It performs binary classification to determine if an image contains a human or not.

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start (Demo Training)

For a quick demonstration with synthetic data:

```bash
python train_model.py
```

This will:
1. Create a model with MobileNetV2 backbone
2. Train on synthetic data (for demonstration)
3. Export to TensorFlow Lite format

## Training with Real Data

### Step 1: Prepare Dataset

**Option A: Download LFW Dataset (Human faces)**
```bash
python download_dataset.py --output_dir ./data
```

**Option B: Use Your Own Data**

Organize your images in this structure:
```
data/
├── train/
│   ├── human/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── non_human/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── validation/
    ├── human/
    │   └── ...
    └── non_human/
        └── ...
```

### Step 2: Train the Model

```bash
python train_with_real_data.py --data_dir ./data --output_dir ./output
```

### Step 3: Use the Model

After training, copy the TFLite model to your Flutter package:

```bash
# For Android
cp output/human_detection_model.tflite ../assets/

# The iOS version will use Core ML (converted separately)
```

## Model Architecture

```
Input: 224x224x3 RGB Image
    ↓
MobileNetV2 (pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense (256 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (1 unit, Sigmoid)
    ↓
Output: Probability [0-1] (human/non-human)
```

## Recommended Datasets

### Human Images (Positive Samples)
- **COCO People**: [cocodataset.org](https://cocodataset.org/) - Filter for person category
- **LFW (Labeled Faces in the Wild)**: [vis-www.cs.umass.edu/lfw](http://vis-www.cs.umass.edu/lfw/)
- **CelebA**: [mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **MPII Human Pose**: [human-pose.mpi-inf.mpg.de](http://human-pose.mpi-inf.mpg.de/)

### Non-Human Images (Negative Samples)
- **Places365**: [places2.csail.mit.edu](http://places2.csail.mit.edu/) - Scene images
- **DTD Textures**: [robots.ox.ac.uk/~vgg/data/dtd](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- **ImageNet**: Select categories without people (nature, objects, etc.)
- **CIFAR-10**: Non-person classes (cars, ships, animals, etc.)

## Output Files

After training, you'll find these files in the `output/` directory:

| File | Description |
|------|-------------|
| `human_detection_model.h5` | Full Keras model |
| `human_detection_model.tflite` | Standard TFLite model |
| `human_detection_model_fp16.tflite` | Float16 optimized (smaller, good accuracy) |
| `human_detection_model_int8.tflite` | Int8 quantized (smallest, requires calibration) |
| `labels.txt` | Class labels |
| `training_history.png` | Training metrics visualization |

## Model Performance Tips

1. **Balanced Dataset**: Ensure roughly equal numbers of human and non-human images
2. **Data Augmentation**: The training script includes rotation, flipping, and zoom
3. **Diverse Images**: Include various poses, lighting conditions, and backgrounds
4. **Negative Mining**: Include challenging non-human images (statues, mannequins, etc.)

## Deployment

The trained model is designed for mobile deployment:

- **Android**: Uses TensorFlow Lite with GPU delegate
- **iOS**: Uses Core ML (or TensorFlow Lite)

Recommended model: `human_detection_model_fp16.tflite` for best size/accuracy balance.
