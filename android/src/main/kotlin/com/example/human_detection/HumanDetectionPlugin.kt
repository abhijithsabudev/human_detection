package com.example.human_detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** HumanDetectionPlugin */
class HumanDetectionPlugin :
    FlutterPlugin,
    MethodCallHandler {
    
    private lateinit var channel: MethodChannel
    private lateinit var context: Context
    
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    private var confidenceThreshold: Float = 0.5f
    private var useGpuDelegate: Boolean = true
    private var numThreads: Int = 4
    private var isObjectDetectionModel: Boolean = true
    
    companion object {
        private const val MODEL_FILE = "human_detection_model.tflite"
        // Object detection models typically use 300x300
        private const val INPUT_SIZE = 300
        private const val PIXEL_SIZE = 3
        // Person class ID in COCO dataset
        private const val PERSON_CLASS_ID = 0
        // Maximum detections
        private const val MAX_DETECTIONS = 10
    }

    override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, "human_detection")
        channel.setMethodCallHandler(this)
        context = flutterPluginBinding.applicationContext
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "getPlatformVersion" -> {
                result.success("Android ${android.os.Build.VERSION.RELEASE}")
            }
            "initialize" -> {
                handleInitialize(call, result)
            }
            "detectHuman" -> {
                handleDetectHuman(call, result)
            }
            "detectHumanFromBytes" -> {
                handleDetectHumanFromBytes(call, result)
            }
            "dispose" -> {
                handleDispose(result)
            }
            else -> {
                result.notImplemented()
            }
        }
    }
    
    private fun handleInitialize(call: MethodCall, result: Result) {
        try {
            val options = call.arguments as? Map<*, *>
            
            confidenceThreshold = (options?.get("confidenceThreshold") as? Double)?.toFloat() ?: 0.5f
            useGpuDelegate = options?.get("useGpuDelegate") as? Boolean ?: true
            numThreads = (options?.get("numThreads") as? Int) ?: 4
            val customModelPath = options?.get("modelPath") as? String
            
            // Load the model
            val modelBuffer = if (customModelPath != null) {
                loadModelFile(customModelPath)
            } else {
                loadModelFromAssets()
            }
            
            // Configure interpreter options
            val interpreterOptions = Interpreter.Options().apply {
                setNumThreads(numThreads)
            }
            
            // Try to use GPU delegate
            if (useGpuDelegate) {
                try {
                    gpuDelegate = GpuDelegate()
                    interpreterOptions.addDelegate(gpuDelegate)
                } catch (e: Exception) {
                    // GPU delegate not available, fall back to CPU
                    gpuDelegate = null
                }
            }
            
            interpreter = Interpreter(modelBuffer, interpreterOptions)
            
            result.success(null)
        } catch (e: Exception) {
            result.error("INITIALIZATION_FAILED", "Failed to initialize: ${e.message}", null)
        }
    }
    
    private fun handleDetectHuman(call: MethodCall, result: Result) {
        try {
            if (interpreter == null) {
                result.error("NOT_INITIALIZED", "Human detection not initialized", null)
                return
            }
            
            val arguments = call.arguments as? Map<*, *>
            val imagePath = arguments?.get("imagePath") as? String
            
            if (imagePath == null) {
                result.error("INVALID_ARGUMENT", "Image path is required", null)
                return
            }
            
            val file = File(imagePath)
            if (!file.exists()) {
                result.error("IMAGE_NOT_FOUND", "Image file not found: $imagePath", null)
                return
            }
            
            val bitmap = BitmapFactory.decodeFile(imagePath)
            if (bitmap == null) {
                result.error("INVALID_IMAGE_FORMAT", "Failed to decode image", null)
                return
            }
            
            val detectionResult = runInference(bitmap)
            bitmap.recycle()
            
            result.success(detectionResult)
        } catch (e: Exception) {
            result.error("INFERENCE_FAILED", "Detection failed: ${e.message}", null)
        }
    }
    
    private fun handleDetectHumanFromBytes(call: MethodCall, result: Result) {
        try {
            if (interpreter == null) {
                result.error("NOT_INITIALIZED", "Human detection not initialized", null)
                return
            }
            
            val arguments = call.arguments as? Map<*, *>
            val imageBytes = arguments?.get("imageBytes") as? ByteArray
            
            if (imageBytes == null) {
                result.error("INVALID_ARGUMENT", "Image bytes are required", null)
                return
            }
            
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            if (bitmap == null) {
                result.error("INVALID_IMAGE_FORMAT", "Failed to decode image bytes", null)
                return
            }
            
            val detectionResult = runInference(bitmap)
            bitmap.recycle()
            
            result.success(detectionResult)
        } catch (e: Exception) {
            result.error("INFERENCE_FAILED", "Detection failed: ${e.message}", null)
        }
    }
    
    private fun runInference(bitmap: Bitmap): Map<String, Any?> {
        val startTime = System.currentTimeMillis()
        
        // Get input tensor shape
        val inputTensor = interpreter?.getInputTensor(0)
        val inputShape = inputTensor?.shape() ?: intArrayOf(1, INPUT_SIZE, INPUT_SIZE, 3)
        val inputHeight = inputShape[1]
        val inputWidth = inputShape[2]
        
        // Preprocess the image
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val inputBuffer = convertBitmapToByteBuffer(scaledBitmap, inputWidth, inputHeight)
        scaledBitmap.recycle()
        
        // Check number of outputs to determine model type
        val numOutputs = interpreter?.outputTensorCount ?: 1
        
        val result = if (numOutputs >= 4) {
            // Object detection model (SSD MobileNet format)
            runObjectDetectionInference(inputBuffer)
        } else {
            // Simple binary classifier
            runBinaryClassifierInference(inputBuffer)
        }
        
        val processingTime = System.currentTimeMillis() - startTime
        return result + mapOf("processingTimeMs" to processingTime.toInt())
    }
    
    private fun runObjectDetectionInference(inputBuffer: ByteBuffer): Map<String, Any?> {
        // Object detection outputs:
        // 0: Bounding boxes [1, N, 4]
        // 1: Class IDs [1, N]
        // 2: Scores [1, N]
        // 3: Number of detections [1]
        
        val numDetections = 10
        val outputLocations = Array(1) { Array(numDetections) { FloatArray(4) } }
        val outputClasses = Array(1) { FloatArray(numDetections) }
        val outputScores = Array(1) { FloatArray(numDetections) }
        val numDetectionsOutput = FloatArray(1)
        
        val outputs = mapOf(
            0 to outputLocations,
            1 to outputClasses,
            2 to outputScores,
            3 to numDetectionsOutput
        )
        
        interpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
        
        // Find person detections (class 0 in COCO)
        var maxPersonConfidence = 0f
        var bestBoundingBox: Map<String, Double>? = null
        
        val actualDetections = numDetectionsOutput[0].toInt().coerceAtMost(numDetections)
        
        for (i in 0 until actualDetections) {
            val classId = outputClasses[0][i].toInt()
            val score = outputScores[0][i]
            
            // Check if it's a person (class 0) with sufficient confidence
            if (classId == PERSON_CLASS_ID && score > maxPersonConfidence) {
                maxPersonConfidence = score
                bestBoundingBox = mapOf(
                    "top" to outputLocations[0][i][0].toDouble(),
                    "left" to outputLocations[0][i][1].toDouble(),
                    "bottom" to outputLocations[0][i][2].toDouble(),
                    "right" to outputLocations[0][i][3].toDouble()
                )
            }
        }
        
        val isHuman = maxPersonConfidence >= confidenceThreshold
        
        return mapOf(
            "isHuman" to isHuman,
            "confidence" to maxPersonConfidence.toDouble(),
            "boundingBox" to bestBoundingBox
        )
    }
    
    private fun runBinaryClassifierInference(inputBuffer: ByteBuffer): Map<String, Any?> {
        // Simple binary classifier with single output
        val outputBuffer = Array(1) { FloatArray(1) }
        interpreter?.run(inputBuffer, outputBuffer)
        
        val confidence = outputBuffer[0][0]
        val isHuman = confidence >= confidenceThreshold
        
        return mapOf(
            "isHuman" to isHuman,
            "confidence" to confidence.toDouble(),
            "boundingBox" to null
        )
    }
    
    private fun convertBitmapToByteBuffer(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        val inputTensor = interpreter?.getInputTensor(0)
        val isQuantized = inputTensor?.dataType()?.name == "UINT8"
        
        val byteBuffer = if (isQuantized) {
            ByteBuffer.allocateDirect(width * height * PIXEL_SIZE)
        } else {
            ByteBuffer.allocateDirect(4 * width * height * PIXEL_SIZE)
        }
        byteBuffer.order(ByteOrder.nativeOrder())
        
        val intValues = IntArray(width * height)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        
        for (pixelValue in intValues) {
            if (isQuantized) {
                // Quantized model expects uint8 values (0-255)
                byteBuffer.put((pixelValue shr 16 and 0xFF).toByte())
                byteBuffer.put((pixelValue shr 8 and 0xFF).toByte())
                byteBuffer.put((pixelValue and 0xFF).toByte())
            } else {
                // Float model expects normalized values
                byteBuffer.putFloat((pixelValue shr 16 and 0xFF) / 255.0f)
                byteBuffer.putFloat((pixelValue shr 8 and 0xFF) / 255.0f)
                byteBuffer.putFloat((pixelValue and 0xFF) / 255.0f)
            }
        }
        
        return byteBuffer
    }
    
    private fun loadModelFromAssets(): MappedByteBuffer {
        val assetManager = context.assets
        val fileDescriptor = assetManager.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val file = File(modelPath)
        val inputStream = FileInputStream(file)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }
    
    private fun handleDispose(result: Result) {
        try {
            interpreter?.close()
            interpreter = null
            
            gpuDelegate?.close()
            gpuDelegate = null
            
            result.success(null)
        } catch (e: Exception) {
            result.error("DISPOSE_FAILED", "Failed to dispose: ${e.message}", null)
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        interpreter?.close()
        gpuDelegate?.close()
    }
}
