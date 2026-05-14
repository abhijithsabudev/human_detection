import Flutter
import UIKit
import TensorFlowLite

public class HumanDetectionPlugin: NSObject, FlutterPlugin {
    
    private var interpreter: Interpreter?
    private var confidenceThreshold: Float = 0.5
    private var numThreads: Int = 4
    
    private let inputSize = 224
    private let pixelSize = 3
    private let imageMean: Float = 127.5
    private let imageStd: Float = 127.5
    
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
        
        // Get output
        let outputTensor = try interpreter?.output(at: 0)
        let outputData = outputTensor?.data
        
        guard let data = outputData else {
            throw NSError(
                domain: "HumanDetection",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Failed to get output data"]
            )
        }
        
        let confidence = data.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> Float in
            pointer.load(as: Float.self)
        }
        
        let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        let isHuman = confidence >= confidenceThreshold
        
        return [
            "isHuman": isHuman,
            "confidence": Double(confidence),
            "processingTimeMs": Int(processingTime)
        ]
    }
    
    private func preprocessImage(_ image: UIImage) -> Data? {
        guard let cgImage = image.cgImage else { return nil }
        
        let targetSize = CGSize(width: inputSize, height: inputSize)
        
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext(),
              let pixelData = resizedImage.cgImage?.dataProvider?.data else {
            return nil
        }
        
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        let bytesPerPixel = 4
        
        var inputData = Data(count: inputSize * inputSize * pixelSize * MemoryLayout<Float>.size)
        
        inputData.withUnsafeMutableBytes { rawBuffer in
            guard let floatBuffer = rawBuffer.bindMemory(to: Float.self).baseAddress else { return }
            
            var pixelIndex = 0
            for y in 0..<inputSize {
                for x in 0..<inputSize {
                    let offset = (y * inputSize + x) * bytesPerPixel
                    
                    let r = Float(data[offset]) 
                    let g = Float(data[offset + 1])
                    let b = Float(data[offset + 2])
                    
                    // Normalize to [-1, 1] for MobileNetV2
                    floatBuffer[pixelIndex] = (r - imageMean) / imageStd
                    floatBuffer[pixelIndex + 1] = (g - imageMean) / imageStd
                    floatBuffer[pixelIndex + 2] = (b - imageMean) / imageStd
                    
                    pixelIndex += 3
                }
            }
        }
        
        return inputData
    }
    
    private func handleDispose(result: @escaping FlutterResult) {
        interpreter = nil
        result(nil)
    }
}
