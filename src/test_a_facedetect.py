"""
TEST A: Face Detection Accuracy Validation

Tests OpenFace's ability to detect faces in ≥95% of video frames
under various quality conditions (resolution, angle, lighting).
"""

import os
import cv2
import pandas as pd
from datetime import datetime
import openface_utils as ofu

# Test configuration
DETECTION_THRESHOLD = 0.95
ENABLE_VISUALIZATION = None  # None = ask user
PLAYBACK_DELAY_MS = 30


def process_video(video_path):
    """Process video through OpenFace and calculate detection metrics."""
    video_name = os.path.basename(video_path).split('.')[0]
    print(f"\n{'='*60}\nProcessing: {video_name}\n{'='*60}")
    
    # Get metadata
    metadata = ofu.get_video_metadata(video_path)
    if not metadata:
        print(f"ERROR: Could not open video file")
        return None
    
    print(f"Resolution: {metadata['resolution']} ({ofu.classify_resolution(metadata['width'], metadata['height'])})")
    print(f"FPS: {metadata['fps']:.2f}, Total Frames: {metadata['total_frames']}")
    
    # Run OpenFace
    try:
        csv_path = ofu.run_openface(video_path)
        df = ofu.load_landmark_data(csv_path, success_only=False)
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    
    # Calculate metrics
    total_frames = metadata['total_frames']
    detected_frames = df[df['success'] == 1].shape[0]
    detection_rate = detected_frames / total_frames if total_frames > 0 else 0
    
    detected_df = df[df['success'] == 1]
    avg_confidence = detected_df['confidence'].mean() if 'confidence' in df.columns and len(detected_df) > 0 else None
    
    passed = detection_rate >= DETECTION_THRESHOLD
    
    # Print results
    print(f"\n{'-'*60}\nDETECTION METRICS:\n{'-'*60}")
    print(f"Total Frames: {total_frames}, Detected: {detected_frames}")
    print(f"Detection Rate: {detection_rate * 100:.2f}%")
    if avg_confidence:
        print(f"Avg Confidence: {avg_confidence:.3f}")
    print(f"{'-'*60}")
    print(f"RESULT: {'✓ PASS' if passed else '✗ FAIL'} (Threshold: {DETECTION_THRESHOLD*100:.0f}%)")
    print(f"{'-'*60}")
    
    return {
        'video_name': video_name,
        'video_path': video_path,
        **metadata,
        'quality_class': ofu.classify_resolution(metadata['width'], metadata['height']),
        'total_frames': total_frames,
        'detected_frames': detected_frames,
        'detection_rate': detection_rate,
        'avg_confidence': avg_confidence,
        'passed': passed,
        'df': df
    }


def visualize_results(results):
    """Visualize detection with bounding boxes and landmarks."""
    if not ENABLE_VISUALIZATION:
        return
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATION - SPACE=Pause, 'q'=Quit, '+/-'=Speed")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(results['video_path'])
    frame_idx, paused, delay_ms = 0, False, PLAYBACK_DELAY_MS
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    df = results['df']
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            row = df[df['frame'] == (frame_idx + 1)]
            
            if not row.empty and row.iloc[0]['success'] == 1.0:
                data = row.iloc[0]
                xs = data[x_cols].values.astype(float)
                ys = data[y_cols].values.astype(float)
                
                # Draw landmarks and box
                ofu.draw_landmarks(frame, xs, ys, show_points=True, show_box=True)
                
                conf = data.get('confidence', 0)
                status = f"Frame {frame_idx}: DETECTED (conf: {conf:.2f})"
                color = (0, 255, 0)
            else:
                status = f"Frame {frame_idx}: NOT DETECTED"
                color = (0, 0, 255)
            
            # Add text overlays
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            ofu.add_text_overlay(frame, [
                (f"Resolution: {results['resolution']}", 70),
                (f"Quality: {results['quality_class']}", 95),
                (f"Speed: {delay_ms}ms/frame", 120)
            ])
            
            current_frame = frame.copy()
            frame_idx += 1
        else:
            frame = current_frame.copy()
            cv2.putText(frame, "PAUSED", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(f"OpenFace Detection - {results['video_name']}", frame)
        
        key = cv2.waitKey(max(1, delay_ms)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in [ord('+'), ord('=')]:
            delay_ms = max(1, delay_ms - 10)
        elif key in [ord('-'), ord('_')]:
            delay_ms = min(500, delay_ms + 10)
    
    cap.release()
    cv2.destroyAllWindows()


def generate_summary_report(all_results):
    """Generate summary report comparing all videos."""
    if not all_results:
        return
    
    print(f"\n{'='*80}\nTEST A: SUMMARY REPORT\n{'='*80}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Detection Threshold: {DETECTION_THRESHOLD*100:.0f}%")
    print(f"Total Videos: {len(all_results)}\n{'-'*80}")
    
    # Summary table
    summary_data = [{
        'Video': r['video_name'],
        'Resolution': r['resolution'],
        'Quality': r['quality_class'],
        'Frames': r['total_frames'],
        'Detected': r['detected_frames'],
        'Rate (%)': f"{r['detection_rate']*100:.2f}",
        'Avg Conf': f"{r['avg_confidence']:.3f}" if r['avg_confidence'] else "N/A",
        'Result': '✓ PASS' if r['passed'] else '✗ FAIL'
    } for r in all_results]
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Statistics
    num_passed = sum(1 for r in all_results if r['passed'])
    print(f"\n{'-'*80}\nOVERALL: {num_passed}/{len(all_results)} passed")
    
    # By quality class
    for quality, name in [('High', '1080p+'), ('Medium', '720p'), ('Low', '480p')]:
        subset = [r for r in all_results if quality in r['quality_class']]
        if subset:
            avg_rate = sum(r['detection_rate'] for r in subset) / len(subset) * 100
            num_pass = sum(1 for r in subset if r['passed'])
            print(f"{quality} Resolution ({name}): {avg_rate:.2f}% ({num_pass}/{len(subset)} passed)")
    
    # Save CSV
    csv_path = os.path.join(ofu.OUTPUT_DIR, f"test_A_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n{'-'*80}\nReport saved: {csv_path}\n{'='*80}\n")


def main():
    """Main execution."""
    global ENABLE_VISUALIZATION
    
    print(f"\n{'='*80}")
    print(f"TEST A: FACE DETECTION ACCURACY VALIDATION")
    print(f"{'='*80}")
    print(f"Purpose: Verify ≥95% face detection across video qualities\n")
    
    # Check OpenFace
    if not ofu.check_openface_binary():
        return
    
    # Find videos
    video_files = ofu.find_videos(['*.mp4', '*.avi', '*.mov', '*.mkv',
                                    '*.MP4', '*.AVI', '*.MOV', '*.MKV'])
    
    if not video_files:
        print(f"ERROR: No videos found in {ofu.VIDEOS_DIR}")
        return
    
    print(f"{'-'*80}\nFound {len(video_files)} video(s):")
    for vf in video_files:
        print(f"  - {os.path.basename(vf)}")
    print(f"{'-'*80}")
    
    # Visualization prompt
    if ENABLE_VISUALIZATION is None:
        choice = input(f"\nShow visualization? (y/N): ").strip().lower()
        ENABLE_VISUALIZATION = (choice == 'y')
    
    print(f"\n{'📹 Visualization enabled' if ENABLE_VISUALIZATION else '⚡ Metrics only'}")
    
    # Process videos
    all_results = []
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n{'='*80}\nVideo {idx}/{len(video_files)}\n{'='*80}")
        result = process_video(video_path)
        if result:
            all_results.append(result)
            if ENABLE_VISUALIZATION:
                visualize_results(result)
    
    # Report
    if all_results:
        generate_summary_report(all_results)
    else:
        print("\n⚠️ No videos processed successfully")


if __name__ == "__main__":
    main()
