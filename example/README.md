# Human Detection Example

A complete example demonstrating the `human_detection` Flutter plugin.

## Features Demonstrated

- Initialize the human detection model
- Pick images from gallery using `image_picker`
- Detect humans in selected images
- Display confidence scores and processing time
- Handle detection results with visual feedback

## Running the Example

1. **Navigate to the example directory:**
   ```bash
   cd example
   ```

2. **Install dependencies:**
   ```bash
   flutter pub get
   ```

3. **Run the app:**
   ```bash
   flutter run
   ```

## Requirements

- Android: API 24+ (Android 7.0)
- iOS: 13.0+
- Flutter: 3.3.0+

## Code Walkthrough

The example app in `lib/main.dart` shows:

1. **Initialization**: How to initialize the `HumanDetection` instance
2. **Image Selection**: Using `image_picker` to select images
3. **Detection**: Calling `detectHuman()` with the image path
4. **Results**: Displaying detection results with confidence scores
5. **Cleanup**: Properly disposing resources

## Screenshot

The app displays:
- A button to select images from gallery
- Detection result (Human/Not Human)
- Confidence percentage
- Processing time in milliseconds
