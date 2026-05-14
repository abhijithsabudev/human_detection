# Contributing to Human Detection

Thank you for your interest in contributing to the Human Detection Flutter plugin! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/abhijithsabudev/human_detection/issues)
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Device/platform information
   - Flutter and Dart version (`flutter doctor -v`)

### Suggesting Features

1. Check existing issues for similar suggestions
2. Create a new issue describing:
   - The feature you'd like to see
   - Why it would be useful
   - Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `flutter test`
5. Ensure code is formatted: `dart format .`
6. Run the analyzer: `flutter analyze`
7. Commit with clear messages: `git commit -m "Add: your feature description"`
8. Push to your fork: `git push origin feature/your-feature-name`
9. Open a Pull Request

### Code Style

- Follow the [Dart Style Guide](https://dart.dev/guides/language/effective-dart/style)
- Use meaningful variable and function names
- Add comments for complex logic
- Include documentation for public APIs

### Testing

- Add tests for new features
- Ensure existing tests pass
- Test on both Android and iOS when possible

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/abhijithsabudev/human_detection.git
   cd human_detection
   ```

2. Install dependencies:
   ```bash
   flutter pub get
   cd example && flutter pub get
   ```

3. Run the example app:
   ```bash
   cd example
   flutter run
   ```

4. Run tests:
   ```bash
   flutter test
   ```

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for helping improve Human Detection! 🙏
