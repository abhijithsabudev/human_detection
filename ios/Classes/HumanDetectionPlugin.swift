import Flutter
import UIKit
import TensorFlowLite

public class HumanDetectionPlugin: NSObject, FlutterPlugin {
    
    private var interpreter: Interpreter?
    private var confidenceThreshold: Float = 0.5
    private var numThreads: Int = 4
    
    // Default input size, will be updated based on model
    private var inputWidth = 300
    private var inputHeight = 300
    private let pixelSize = 3
    
    // Person class ID in COCO dataset
    private let personClassId = 0
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "human_detection", binaryMessenger: registrar.messenger())
        let instance = HumanDetectionPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "getPlatformVersion":
            result("iOS " + UIDevice.current.systemVersion)
            
        case "initialize":
            handleInitialize(call: call, result: result)
            
        case "detectHuman":
            handleDetectHuman(call: call, result: result)
            
        case "detectHumanFromBytes":
            handleDetectHumanFromBytes(call: call, result: result)
            
        case "dispose":
            handleDispose(result: result)
            
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    private func handleInitialize(call: FlutterMethodCall, result: @escaping FlutterResult) {
        do {
            if let options = call.arguments as? [String: Any] {
                confidenceThreshold = Float(options["confidenceThreshold"] as? Double ?? 0.5)
                numThreads = options["numThreads"] as? Int ?? 4
                
                if let customModelPath = options["modelPath"] as? String {
                    try loadModel(from: customModelPath)
                } else {
                    try loadModelFromBundle()
                }
            } else {
                try loadModelFromBundle()
            }
            
            // Get input tensor shape
            if let inputTensor = try? interpreter?.input(at: 0) {
                let shape = inputTensor.shape.dimensions
                if shape.count >= 3 {
                    inputHeight = shape[1]
                    inputWidth = shape[2]
                }
            }
            
            result(nil)
        } catch {
            result(FlutterError(
                code: "INITIALIZATION_FAILED",
                message: "Failed to initialize: \(error.localizedDescription)",
                details: nil
            ))
        }
    }
    
    private func loadModelFromBundle() throws {
        guard let modelPath = Bundle(for: type(of: self)).path(
            forResource: "human_detection_model",
            ofType: "tflite"
        ) else {
            throw NSError(
                domain: "HumanDetection",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Model file not found in bundle"]
            )
        }
        
        try loadModel(from: modelPath)
    }
    
    private func loadModel(from path: String) throws {
        var options = Interpreter.Options()
        options.threadCount = numThreads
        
        interpreter = try Interpreter(modelPath: path, options: options)
        try interpreter?.allocateTensors()
    }
    
    private func handleDetectHuman(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard interpreter != nil else {
            result(FlutterError(
                code: "NOT_INITIALIZED",
                message: "Human detection not initialized",
                details: nil
            ))
            return
        }
        
        guard let arguments = call.arguments as? [String: Any],
              let imagePath = arguments["imagePath"] as? String else {
            result(FlutterError(
                code: "INVALID_ARGUMENT",
                message: "Image path is required",
                details: nil
            ))
            return
        }
        
        guard let image = UIImage(contentsOfFile: imagePath) else {
            result(FlutterError(
                code: "IMAGE_NOT_FOUND",
                message: "Failed to load image from path",
                details: nil
            ))
            return
        }
        
        do {
            let detectionResult = try runInference(on: image)
            result(detectionResult)
        } catch {
            result(FlutterError(
                code: "INFERENCE_FAILED",
                message: "Detection failed: \(error.localizedDescription)",
                details: nil
            ))
        }
    }
    
    private func handleDetectHumanFromBytes(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard interpreter != nil else {
            result(FlutterError(
                code: "NOT_INITIALIZED",
                message: "Human detection not initialized",
                details: nil
            ))
            return
        }
        
        guard let arguments = call.arguments as? [String: Any],
              let imageBytes = arguments["imageBytes"] as? FlutterStandardTypedData else {
            result(FlutterError(
                code: "INVALID_ARGUMENT",
                message: "Image bytes are required",
                details: nil
            ))
            return
        }
        
        guard let image = UIImage(data: imageBytes.data) else {
            result(FlutterError(
                code: "INVALID_IMAGE_FORMAT",
                message: "Failed to decode image bytes",
                details: nil
            ))
            return
        }
        
        do {
            let detectionResult = try runInference(on: image)
            result(detectionResult)
        } catch {
            result(FlutterError(
                code: "INFERENCE_FAILED",
                message: "Detection failed: \(error.localizedDescription)",
                details: nil
            ))
        }
    }
    
    private func runInference(on image: UIImage) throws -> [String: Any] {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Preprocess the image
        guard let inputData = preprocessImage(image) else {
            throw NSError(
                domain: "HumanDetection",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to preprocess image"]
            )
        }
        
        // Run inference
        try interpreter?.copy(inputData, toInputAt: 0)
        try interpreter?.invoke()
        
        // Check number of outputs to determine model type
        let outputCount = interpreter?.outputTensorCount ?? 1
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        if outputCount >= 4 {
            // Object detection model (SSD MobileNet format)
            return try runObjectDetectionInference(processingTime: processingTime)
        } else {
            // Binary classifier
            return try runBinaryClassifierInference(processingTime: processingTime)
        }
    }
    
    private func runObjectDetectionInference(processingTime: Double) throws -> [String: Any] {
        // Object detection outputs:
        // 0: Bounding boxes [1, N, 4]
        // 1: Class IDs [1, N]
        // 2: Scores [1, N]
        // 3: Number of detections [1]
        
        guard let boxesTensor = try interpreter?.output(at: 0),
              let classesTensor = try interpreter?.output(at: 1),
              let scoresTensor = try interpreter?.output(at: 2),
              let numDetectionsTensor = try interpreter?.output(at: 3) else {
            throw NSError(
                domain: "HumanDetection",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Failed to get output tensors"]
            )
        }
        
        let numDetections = numDetectionsTensor.data.withUnsafeBytes { ptr -> Int in
            Int(ptr.load(as: Float.self))
        }
        
        var maxPersonConfidence: Float = 0
        var bestBoundingBox: [String: Double]? = nil
        
        let maxDetections = min(numDetections, 10)
        
        for i in 0..<maxDetections {
            let classId = classesTensor.data.withUnsafeBytes { ptr -> Int in
                Int(ptr.load(fromByteOffset: i * MemoryLayout<Float>.size, as: Float.self))
            }
            
            let score = scoresTensor.data.withUnsafeBytes { ptr -> Float in
                ptr.load(fromByteOffset: i * MemoryLayout<Float>.size, as: Float.self)
            }
            
            // Check if it's a person (class 0) with sufficient confidence
            if classId == personClassId && score > maxPersonConfidence {
                maxPersonConfidence = score
                
                let boxOffset = i * 4 * MemoryLayout<Float>.size
                bestBoundingBox = boxesTensor.data.withUnsafeBytes { ptr -> [String: Double] in
                    [
                        "top": Double(ptr.load(fromByteOffset: boxOffset, as: Float.self)),
                        "left": Double(ptr.load(fromByteOffset: boxOffset + MemoryLayout<Float>.size, as: Float.self)),
                        "bottom": Double(ptr.load(fromByteOffset: boxOffset + 2 * MemoryLayout<Float>.size, as: Float.self)),
                        "right": Double(ptr.load(fromByteOffset: boxOffset + 3 * MemoryLayout<Float>.size, as: Float.self))
                    ]
                }
            }
        }
        
        let isHuman = maxPersonConfidence >= confidenceThreshold
        
        return [
            "isHuman": isHuman,
            "confidence": Double(maxPersonConfidence),
            "processingTimeMs": Int(processingTime),
            "boundingBox": bestBoundingBox as Any
        ]
    }
    
    private func runBinaryClassifierInference(processingTime: Double) throws -> [String: Any] {
        guard let outputTensor = try interpreter?.output(at: 0) else {
            throw NSError(
                domain: "HumanDetection",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Failed to get output tensor"]
            )
        }
        
        let confidence = outputTensor.data.withUnsafeBytes { ptr -> Float in
            ptr.load(as: Float.self)
        }
        
        let isHuman = confidence >= confidenceThreshold
        
        return [
            "isHuman": isHuman,
            "confidence": Double(confidence),
            "processingTimeMs": Int(processingTime),
            "boundingBox": NSNull()
        ]
    }
    
    private func preprocessImage(_ image: UIImage) -> Data? {
        let targetSize = CGSize(width: inputWidth, height: inputHeight)
        
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext(),
              let pixelData = resizedImage.cgImage?.dataProvider?.data else {
            return nil
        }
        
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        let bytesPerPixel = 4
        
        // Check if model expects quantized (uint8) or float input
        let inputTensor = try? interpreter?.input(at: 0)
        let isQuantized = inputTensor?.dataType == .uInt8
        
        if isQuantized {
            // Quantized model expects uint8 values (0-255)
            var inputData = Data(count: inputWidth * inputHeight * pixelSize)
            
            inputData.withUnsafeMutableBytes { rawBuffer in
                guard let byteBuffer = rawBuffer.bindMemory(to: UInt8.self).baseAddress else { return }
                
                var pixelIndex = 0
                for y in 0..<inputHeight {
                    for x in 0..<inputWidth {
                        let offset = (y * inputWidth + x) * bytesPerPixel
                        
                        byteBuffer[pixelIndex] = data[offset]     // R
                        byteBuffer[pixelIndex + 1] = data[offset + 1] // G
                        byteBuffer[pixelIndex + 2] = data[offset + 2] // B
                        
                        pixelIndex += 3
                    }
                }
            }
            
            return inputData
        } else {
            // Float model expects normalized values (0-1)
            var inputData = Data(count: inputWidth * inputHeight * pixelSize * MemoryLayout<Float>.size)
            
            inputData.withUnsafeMutableBytes { rawBuffer in
                guard let floatBuffer = rawBuffer.bindMemory(to: Float.self).baseAddress else { return }
                
                var pixelIndex = 0
                for y in 0..<inputHeight {
                    for x in 0..<inputWidth {
                        let offset = (y * inputWidth + x) * bytesPerPixel
                        
                        floatBuffer[pixelIndex] = Float(data[offset]) / 255.0
                        floatBuffer[pixelIndex + 1] = Float(data[offset + 1]) / 255.0
                        floatBuffer[pixelIndex + 2] = Float(data[offset + 2]) / 255.0
                        
                        pixelIndex += 3
                    }
                }
            }
            
            return inputData
        }
    }
    
    private func handleDispose(result: @escaping FlutterResult) {
        interpreter = nil
        result(nil)
    }
}
