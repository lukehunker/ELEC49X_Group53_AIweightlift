import os
import cv2
import subprocess
import pandas as pd
import numpy as np
import glob
from datetime import datetime

# =========================================
# USER CONFIGURATION
# =========================================
# Get the script directory and build relative paths from repository root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from src/ to repo root

# Path to OpenFace FeatureExtraction binary (relative to repo root)
OPENFACE_BIN = os.path.join(REPO_ROOT, "OpenFace", "build", "bin", "FeatureExtraction")

# Directory containing test videos (relative to repo root)
VIDEOS_DIR = os.path.join(REPO_ROOT, "videos")

# Output directory for OpenFace results (relative to repo root)
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visualization enabled (set to False to skip visualization and just get metrics)
ENABLE_VISUALIZATION = True

# Visualization playback speed (milliseconds per frame, higher = slower)
# 30ms = ~33 fps, 100ms = 10 fps, 0 = wait for key press
PLAYBACK_DELAY_MS = 30

# Detection rate threshold for pass/fail
DETECTION_THRESHOLD = 0.95


# =========================================
# HELPER FUNCTIONS
# =========================================
def get_video_metadata(video_path):
    """Extract video metadata including resolution, FPS, and frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    metadata = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'resolution': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    }
    cap.release()
    return metadata


def classify_resolution(width, height):
    """Classify video resolution quality."""
    pixels = width * height
    if pixels >= 1920 * 1080:
        return "High (1080p+)"
    elif pixels >= 1280 * 720:
        return "Medium (720p)"
    elif pixels >= 640 * 480:
        return "Low (480p)"
    else:
        return "Very Low (<480p)"


def process_video(video_path):
    """
    Process a single video through OpenFace and calculate detection metrics.
    Returns a dictionary with all relevant metrics.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Get video metadata
    metadata = get_video_metadata(video_path)
    if metadata is None:
        print(f"ERROR: Could not open video file: {video_path}")
        return None
    
    print(f"Resolution: {metadata['resolution']} ({metadata['width']}x{metadata['height']})")
    print(f"Quality Class: {classify_resolution(metadata['width'], metadata['height'])}")
    print(f"FPS: {metadata['fps']:.2f}")
    print(f"Total Frames: {metadata['total_frames']}")
    
    # Run OpenFace if CSV doesn't exist
    csv_pattern = os.path.join(OUTPUT_DIR, f"{video_name}*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"\nRunning OpenFace on {video_name}...")
        if not os.path.isfile(OPENFACE_BIN):
            raise FileNotFoundError(f"OpenFace binary NOT found at: {OPENFACE_BIN}")
        
        # Run FeatureExtraction with 2D landmarks (-2Dfp) and AUs (-aus)
        cmd = [OPENFACE_BIN, "-f", video_path, "-out_dir", OUTPUT_DIR, "-2Dfp", "-aus"]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: OpenFace processing failed: {e}")
            return None
        
        csv_files = glob.glob(csv_pattern)
        if not csv_files:
            print("ERROR: OpenFace finished but did not generate a CSV file.")
            return None
    else:
        print(f"Using existing OpenFace output: {os.path.basename(csv_files[0])}")
    
    csv_path = csv_files[0]
    
    # Load OpenFace results
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        return None
    
    # Calculate detection metrics
    total_frames_video = metadata['total_frames']
    detected_frames = df[df['success'] == 1].shape[0]
    detection_rate = detected_frames / total_frames_video if total_frames_video > 0 else 0
    
    # Calculate confidence statistics for detected frames
    detected_df = df[df['success'] == 1]
    avg_confidence = detected_df['confidence'].mean() if 'confidence' in df.columns and len(detected_df) > 0 else None
    min_confidence = detected_df['confidence'].min() if 'confidence' in df.columns and len(detected_df) > 0 else None
    
    # Pass/Fail determination
    passed = detection_rate >= DETECTION_THRESHOLD
    
    # Print results
    print(f"\n{'-'*60}")
    print(f"DETECTION METRICS:")
    print(f"{'-'*60}")
    print(f"Total Video Frames:        {total_frames_video}")
    print(f"OpenFace Detected Frames:  {detected_frames}")
    print(f"Detection Rate:            {detection_rate * 100:.2f}%")
    if avg_confidence is not None:
        print(f"Average Confidence:        {avg_confidence:.3f}")
        print(f"Min Confidence:            {min_confidence:.3f}")
    print(f"{'-'*60}")
    print(f"RESULT: {'✓ PASS' if passed else '✗ FAIL'} (Threshold: {DETECTION_THRESHOLD*100:.0f}%)")
    print(f"{'-'*60}")
    
    # Compile results
    results = {
        'video_name': video_name,
        'video_path': video_path,
        'resolution': metadata['resolution'],
        'width': metadata['width'],
        'height': metadata['height'],
        'quality_class': classify_resolution(metadata['width'], metadata['height']),
        'fps': metadata['fps'],
        'total_frames': total_frames_video,
        'detected_frames': detected_frames,
        'detection_rate': detection_rate,
        'avg_confidence': avg_confidence,
        'min_confidence': min_confidence,
        'passed': passed,
        'csv_path': csv_path,
        'df': df
    }
    
    return results


def visualize_results(results):
    """Visualize detection results with bounding boxes."""
    if not ENABLE_VISUALIZATION:
        return
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATION MODE - Controls:")
    print(f"  SPACE = Pause/Resume")
    print(f"  'q' = Quit visualization")
    print(f"  '+' = Speed up")
    print(f"  '-' = Slow down")
    print(f"{'='*60}")
    
    video_path = results['video_path']
    df = results['df']
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    paused = False
    delay_ms = PLAYBACK_DELAY_MS
    
    # Identify landmark columns for bounding box calculation
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Find corresponding row in OpenFace CSV (1-based index)
            row = df[df['frame'] == (frame_idx + 1)]
            
            detected = False
            if not row.empty and row.iloc[0]['success'] == 1.0:
                detected = True
                data = row.iloc[0]
                
                # Extract all 68 landmark coordinates to find the bounding box
                xs = data[x_cols].values.astype(float)
                ys = data[y_cols].values.astype(float)
                
                # Calculate bounding box from min/max landmarks
                x_min, x_max = int(np.min(xs)), int(np.max(xs))
                y_min, y_max = int(np.min(ys)), int(np.max(ys))
                
                # Add padding to bounding box for better visibility
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame.shape[1], x_max + padding)
                y_max = min(frame.shape[0], y_max + padding)
                
                # Draw Green Bounding Box (thicker for better visibility)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                
                # Draw individual landmarks (optional, shows detail)
                for i in range(68):
                    x, y = int(xs[i]), int(ys[i])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                # Draw status text with confidence
                confidence = data.get('confidence', 0)
                status_text = f"Frame {frame_idx}: DETECTED (conf: {confidence:.2f})"
                cv2.putText(frame, status_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Draw Red status text if not detected
                cv2.putText(frame, f"Frame {frame_idx}: NOT DETECTED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add info overlay with dark background for readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 50), (400, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Add resolution and playback info
            res_text = f"Resolution: {results['resolution']}"
            cv2.putText(frame, res_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            quality_text = f"Quality: {results['quality_class']}"
            cv2.putText(frame, quality_text, (10, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            speed_text = f"Speed: {delay_ms}ms/frame"
            cv2.putText(frame, speed_text, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Store current frame for paused display
            current_frame = frame.copy()
            frame_idx += 1
        else:
            # Display paused message
            frame = current_frame.copy()
            cv2.putText(frame, "PAUSED - Press SPACE to resume", (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow(f"OpenFace Detection - {results['video_name']}", frame)
        
        # Wait for key press with configurable delay
        key = cv2.waitKey(max(1, delay_ms)) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar to pause/resume
            paused = not paused
        elif key == ord('+') or key == ord('='):  # Speed up (decrease delay)
            delay_ms = max(1, delay_ms - 10)
            print(f"Playback speed: {delay_ms}ms/frame")
        elif key == ord('-') or key == ord('_'):  # Slow down (increase delay)
            delay_ms = min(500, delay_ms + 10)
            print(f"Playback speed: {delay_ms}ms/frame")
    
    cap.release()
    cv2.destroyAllWindows()


def generate_summary_report(all_results):
    """Generate a summary report comparing all processed videos."""
    if not all_results:
        print("\nNo results to summarize.")
        return
    
    print(f"\n{'='*80}")
    print(f"TEST A: INPUT DATA QUALITY VALIDATION - SUMMARY REPORT")
    print(f"{'='*80}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Detection Threshold: {DETECTION_THRESHOLD*100:.0f}%")
    print(f"\n{'-'*80}")
    
    # Create summary table
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Video': result['video_name'],
            'Resolution': result['resolution'],
            'Quality': result['quality_class'],
            'Frames': result['total_frames'],
            'Detected': result['detected_frames'],
            'Rate (%)': f"{result['detection_rate']*100:.2f}",
            'Avg Conf': f"{result['avg_confidence']:.3f}" if result['avg_confidence'] is not None else "N/A",
            'Result': '✓ PASS' if result['passed'] else '✗ FAIL'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print(f"\n{'-'*80}")
    print(f"OVERALL STATISTICS:")
    print(f"{'-'*80}")
    
    # Calculate statistics by quality class
    high_res = [r for r in all_results if 'High' in r['quality_class']]
    medium_res = [r for r in all_results if 'Medium' in r['quality_class']]
    low_res = [r for r in all_results if 'Low' in r['quality_class'] and 'Very' not in r['quality_class']]
    
    if high_res:
        avg_rate = np.mean([r['detection_rate'] for r in high_res]) * 100
        print(f"High Resolution (1080p+):  Avg Detection Rate = {avg_rate:.2f}%")
    
    if medium_res:
        avg_rate = np.mean([r['detection_rate'] for r in medium_res]) * 100
        print(f"Medium Resolution (720p):  Avg Detection Rate = {avg_rate:.2f}%")
    
    if low_res:
        avg_rate = np.mean([r['detection_rate'] for r in low_res]) * 100
        print(f"Low Resolution (480p):     Avg Detection Rate = {avg_rate:.2f}%")
    
    # Save summary to CSV
    summary_path = os.path.join(OUTPUT_DIR, f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'-'*80}")
    print(f"Summary report saved to: {summary_path}")
    print(f"{'='*80}\n")


# =========================================
# MAIN EXECUTION
# =========================================
def main():
    """Main function to process all videos and generate report."""
    print(f"\n{'='*80}")
    print(f"OpenFace Input Data Quality Validation Pipeline")
    print(f"{'='*80}")
    
    # Check if OpenFace binary exists
    if not os.path.isfile(OPENFACE_BIN):
        print(f"ERROR: OpenFace binary NOT found at: {OPENFACE_BIN}")
        print("Please ensure OpenFace is built and the path is correct.")
        return
    
    # Find all video files in the videos directory
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV', '*.MKV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(VIDEOS_DIR, ext)))
    
    if not video_files:
        print(f"ERROR: No video files found in: {VIDEOS_DIR}")
        print("Please add video files to the videos directory.")
        return
    
    print(f"\nFound {len(video_files)} video file(s) to process:")
    for vf in video_files:
        print(f"  - {os.path.basename(vf)}")
    
    # Process each video
    all_results = []
    for video_path in sorted(video_files):
        result = process_video(video_path)
        if result is not None:
            all_results.append(result)
            
            # Visualize if enabled
            if ENABLE_VISUALIZATION:
                visualize_results(result)
    
    # Generate summary report
    generate_summary_report(all_results)


if __name__ == "__main__":
    main()
