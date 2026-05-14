#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint human_detection.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'human_detection'
  s.version          = '0.0.1'
  s.summary          = 'A Flutter plugin for detecting humans in images using machine learning.'
  s.description      = <<-DESC
A Flutter plugin for detecting humans in images using TensorFlow Lite.
Provides a simple API to determine if an image contains a human with confidence scores.
                       DESC
  s.homepage         = 'https://github.com/example/human_detection'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.dependency 'TensorFlowLiteSwift', '~> 2.14.0'
  s.platform = :ios, '13.0'
  
  # Include the model file
  s.resources = ['Assets/**/*']

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 
    'DEFINES_MODULE' => 'YES', 
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' 
  }
  s.swift_version = '5.0'

  # Privacy manifest
  s.resource_bundles = {'human_detection_privacy' => ['Resources/PrivacyInfo.xcprivacy']}
end
