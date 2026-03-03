#!/usr/bin/env python3
"""
Test the API server with a sample video.

Usage:
    python test_api.py path/to/video.mp4 "Squat"
"""
import requests
import sys
import os

def test_health():
    """Test if server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
            print(f"   {response.json()}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        print("   Start server with: cd src && python api_server.py")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_prediction(video_path, lift_type="Squat"):
    """Test RPE prediction"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    print(f"\n📤 Uploading video: {os.path.basename(video_path)}")
    print(f"   Lift type: {lift_type}")
    print(f"   File size: {os.path.getsize(video_path) / 1024 / 1024:.2f} MB")
    print("\n   Processing (this may take 30-60 seconds)...")
    
    try:
        with open(video_path, 'rb') as f:
            files = {'video': f}
            data = {'lift_type': lift_type}
            
            response = requests.post(
                "http://localhost:8000/predict",
                files=files,
                data=data,
                timeout=300  # 5 minute timeout
            )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Prediction successful!")
            print(f"   🏋️ Predicted RPE: {result['predicted_rpe']}")
            print(f"   📊 Confidence: {result['confidence']*100:.0f}%")
            print(f"   🎬 Video: {result['video_name']}")
            
            if result.get('warnings'):
                print(f"\n   ⚠️  Warnings:")
                for warning in result['warnings']:
                    print(f"      - {warning}")
            
            # Show features extracted
            features = result.get('features', {})
            print(f"\n   📋 Features extracted:")
            print(f"      OpenFace: {len(features.get('openface_features', {}))} features")
            print(f"      Pose: {len(features.get('pose_features', {}))} features")
            print(f"      Bar Speed: {len(features.get('bar_features', {}))} features")
            print(f"      Detection rate: {features.get('detection_rate', 0)*100:.1f}%")
            
            return True
        else:
            print(f"\n❌ Prediction failed: {response.status_code}")
            print(f"   {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("\n❌ Request timeout - video processing took too long")
        return False
    except Exception as e:
        print(f"\n❌ Prediction error: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("API Server Test")
    print("="*60)
    
    # Test health
    print("\n1. Testing server health...")
    if not test_health():
        sys.exit(1)
    
    # Test prediction
    if len(sys.argv) < 2:
        print("\n⚠️  No video provided for prediction test")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} path/to/video.mp4 [lift_type]")
        print("\nExample:")
        print(f"  python {sys.argv[0]} ../lifting_videos/Demo_Videos/Squat_1.mp4 Squat")
        sys.exit(0)
    
    video_path = sys.argv[1]
    lift_type = sys.argv[2] if len(sys.argv) > 2 else "Squat"
    
    print("\n2. Testing RPE prediction...")
    test_prediction(video_path, lift_type)
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
