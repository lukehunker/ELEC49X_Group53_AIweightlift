import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

class RPEApiService {
  // API URL Configuration:
  // - Android Emulator: 'http://10.0.2.2:8000' (emulator's special address for host)
  // - Windows Desktop: 'http://localhost:8000'
  // - Deployed: 'https://your-render-app.onrender.com'
  static const String baseUrl = String.fromEnvironment('API_BASE_URL', defaultValue: 'http://localhost:8000');
  
  /// Check if the API server is healthy
  static Future<bool> checkHealth() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/health'),
      ).timeout(const Duration(seconds: 5));
      
      return response.statusCode == 200;
    } catch (e) {
      print('Health check failed: $e');
      return false;
    }
  }
  
  /// Predict RPE from a video file
  /// 
  /// Parameters:
  /// - videoPath: Path to the video file
  /// - liftType: One of 'squat', 'bench', or 'deadlift'
  /// 
  /// Returns a Map with prediction results or throws an exception
  static Future<Map<String, dynamic>> predictRPE({
    required String videoPath,
    required String liftType,
  }) async {
    try {
      // Map lift type to backend format
      final liftTypeMap = {
        'squat': 'Squat',
        'bench press': 'Bench Press',
        'bench': 'Bench Press',
        'deadlift': 'Deadlift',
      };
      
      final normalizedInput = liftType.toLowerCase();
      final backendLiftType = liftTypeMap[normalizedInput];
      
      if (backendLiftType == null) {
        throw Exception('Invalid lift type: $liftType. Must be Squat, Bench Press, or Deadlift');
      }
      
      // Create multipart request
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/predict'),
      );
      
      // Add video file
      request.files.add(
        await http.MultipartFile.fromPath(
          'video',
          videoPath,
        ),
      );
      
      // Add lift type as form field
      request.fields['lift_type'] = backendLiftType;
      
      print('Sending request to $baseUrl/predict');
      print('Lift type: $backendLiftType');
      print('Video: $videoPath');
      
      // Send request with timeout (videos take 3-5 minutes to process)
      final streamedResponse = await request.send().timeout(
        const Duration(minutes: 10),
        onTimeout: () {
          throw Exception('Request timed out after 10 minutes');
        },
      );
      
      // Get response
      final response = await http.Response.fromStream(streamedResponse);
      
      print('Response status: ${response.statusCode}');
      print('Response body: ${response.body}');
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data;
      } else {
        final error = json.decode(response.body);
        throw Exception('API Error: ${error['detail'] ?? 'Unknown error'}');
      }
      
    } on SocketException {
      throw Exception('Cannot connect to server. Make sure the API is running at $baseUrl');
    } on http.ClientException {
      throw Exception('Network error. Check your connection.');
    } catch (e) {
      throw Exception('Prediction failed: $e');
    }
  }
  
  /// Get user-friendly lift type name
  static String getLiftTypeName(String liftType) {
    switch (liftType.toLowerCase()) {
      case 'squat':
        return 'Squat';
      case 'bench':
        return 'Bench Press';
      case 'deadlift':
        return 'Deadlift';
      default:
        return liftType;
    }
  }
  
  /// Convert backend lift type to display name
  static String normalizeApiLiftType(String apiType) {
    switch (apiType.toLowerCase()) {
      case 'squat':
        return 'Squat';
      case 'bench':
        return 'Bench Press';
      case 'deadlift':
        return 'Deadlift';
      default:
        return apiType;
    }
  }
}
