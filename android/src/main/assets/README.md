# Human Detection Model

This directory should contain the TensorFlow Lite model file:
- `human_detection_model.tflite`

## Generating the Model

To generate the model, run the Python script:

```bash
cd model_training
pip install -r requirements.txt
python generate_minimal_model.py
```

This will create a minimal model for testing. For production, train a proper model using:

```bash
python train_with_real_data.py --data_dir ./data
```

## Model Format

The model should:
- Accept input: `[1, 224, 224, 3]` (batch, height, width, channels)
- Output: `[1, 1]` (single sigmoid probability)
- Use float32 input/output
