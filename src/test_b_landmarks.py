"""
TEST B: Facial Landmark Tracking Validation

Tests OpenFace's landmark tracking stability and accuracy:
1. Stability Test: Coordinate variation during neutral/still expressions (≤2px)
2. Dynamic Test: Landmark movement during exaggerated expressions (smooth tracking)
"""

import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import openface_utils as ofu

# Test configuration
STABILITY_THRESHOLD = 2.0
ENABLE_VIDEO_VISUALIZATION = True
PLAYBACK_DELAY_MS = 30
SMOOTHNESS_THRESHOLD = 0.3

# Landmark pairs for distance measurement (expressions)
LANDMARK_PAIRS = {
    'eyebrow_distance': (19, 24),
    'eye_height_left': (37, 41),
    'eye_height_right': (43, 47),
    'mouth_height': (51, 57),
    'mouth_width': (48, 54),
}


# =========================================
# STABILITY TEST
# =========================================
def stability_test(df, video_name):
    """Test landmark stability for neutral/still expression."""
    print(f"\n{'='*60}\nSTABILITY TEST: {video_name}\n{'='*60}")
    
    stability_data = []
    max_std = 0.0
    
    for landmark_idx in range(68):
        x_col, y_col = f'x_{landmark_idx}', f'y_{landmark_idx}'
        if x_col not in df.columns or y_col not in df.columns:
            continue
        
        x_std = np.std(df[x_col].values)
        y_std = np.std(df[y_col].values)
        combined_std = np.sqrt(x_std**2 + y_std**2)
        
        stability_data.append({
            'landmark': landmark_idx,
            'x_std': x_std,
            'y_std': y_std,
            'combined_std': combined_std,
            'passed': combined_std <= STABILITY_THRESHOLD
        })
        max_std = max(max_std, combined_std)
    
    stability_df = pd.DataFrame(stability_data)
    failed_landmarks = stability_df[~stability_df['passed']]
    passed = len(failed_landmarks) == 0
    
    # Print summary
    print(f"\nTotal Landmarks: {len(stability_df)}")
    print(f"Threshold: {STABILITY_THRESHOLD} pixels")
    print(f"Max Std Dev: {max_std:.3f} pixels")
    print(f"Failed: {len(failed_landmarks)}/{len(stability_df)}")
    
    if len(failed_landmarks) > 0:
        print(f"\nLandmarks exceeding threshold:")
        for _, row in failed_landmarks.head(10).iterrows():
            print(f"  Landmark {row['landmark']}: {row['combined_std']:.3f}px")
    
    print(f"\n{'-'*60}")
    print(f"RESULT: {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"{'-'*60}")
    
    # Save report
    report_path = os.path.join(ofu.OUTPUT_DIR, f"{video_name}_stability.csv")
    stability_df.to_csv(report_path, index=False)
    print(f"Report saved: {report_path}")
    
    return {
        'passed': passed,
        'max_std': max_std,
        'num_failed': len(failed_landmarks),
        'num_total': len(stability_df),
        'landmark_stats': {i: {'combined_std': row['combined_std']} 
                          for i, row in stability_df.iterrows()}
    }, stability_df


def plot_stability_heatmap(stability_df, video_name):
    """Create stability visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    landmarks = stability_df['landmark'].values
    x_stds = stability_df['x_std'].values
    y_stds = stability_df['y_std'].values
    
    colors1 = ['green' if x <= STABILITY_THRESHOLD else 'red' for x in x_stds]
    ax1.bar(landmarks, x_stds, color=colors1, alpha=0.7)
    ax1.axhline(y=STABILITY_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({STABILITY_THRESHOLD}px)')
    ax1.set_xlabel('Landmark Index')
    ax1.set_ylabel('Standard Deviation (pixels)')
    ax1.set_title(f'X-Coordinate Stability - {video_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    colors2 = ['green' if y <= STABILITY_THRESHOLD else 'red' for y in y_stds]
    ax2.bar(landmarks, y_stds, color=colors2, alpha=0.7)
    ax2.axhline(y=STABILITY_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({STABILITY_THRESHOLD}px)')
    ax2.set_xlabel('Landmark Index')
    ax2.set_ylabel('Standard Deviation (pixels)')
    ax2.set_title(f'Y-Coordinate Stability - {video_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(ofu.OUTPUT_DIR, f"{video_name}_stability_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved: {plot_path}")
    plt.close()


# =========================================
# DYNAMIC TEST
# =========================================
def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def dynamic_test(df, video_name):
    """Test landmark tracking during dynamic expressions."""
    print(f"\n{'='*60}\nDYNAMIC TEST: {video_name}\n{'='*60}")
    
    pair_metrics = {}
    smoothness_scores = {}
    
    for pair_name, (idx1, idx2) in LANDMARK_PAIRS.items():
        x1_col, y1_col = f'x_{idx1}', f'y_{idx1}'
        x2_col, y2_col = f'x_{idx2}', f'y_{idx2}'
        
        distances = np.array([
            calculate_distance(row[x1_col], row[y1_col], row[x2_col], row[y2_col])
            for _, row in df.iterrows()
        ])
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        range_dist = max_dist - min_dist
        
        # Smoothness (velocity std deviation)
        velocities = np.abs(np.diff(distances))
        smoothness = np.std(velocities)
        normalized_smoothness = smoothness / range_dist if range_dist > 0 else 0
        
        pair_metrics[pair_name] = {
            'mean': mean_dist,
            'std': std_dist,
            'min': min_dist,
            'max': max_dist,
            'range': range_dist,
            'smoothness': smoothness,
            'normalized_smoothness': normalized_smoothness,
            'distances': distances
        }
        smoothness_scores[pair_name] = normalized_smoothness
        
        print(f"\n{pair_name}:")
        print(f"  Range: {min_dist:.2f} - {max_dist:.2f}px (Δ={range_dist:.2f})")
        print(f"  Mean: {mean_dist:.2f} ± {std_dist:.2f}px")
        print(f"  Smoothness: {normalized_smoothness:.4f}")
    
    avg_smoothness = np.mean(list(smoothness_scores.values()))
    passed = avg_smoothness < SMOOTHNESS_THRESHOLD
    
    print(f"\n{'-'*60}")
    print(f"Avg Smoothness: {avg_smoothness:.4f} (threshold: {SMOOTHNESS_THRESHOLD})")
    print(f"RESULT: {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"{'-'*60}")
    
    return {
        'passed': passed,
        'pair_metrics': pair_metrics,
        'smoothness_scores': smoothness_scores,
        'avg_smoothness': avg_smoothness
    }


def plot_dynamic_tracking(results, video_name):
    """Plot landmark distances over time."""
    pair_metrics = results['pair_metrics']
    num_pairs = len(pair_metrics)
    
    fig, axes = plt.subplots(num_pairs, 1, figsize=(12, 3 * num_pairs))
    if num_pairs == 1:
        axes = [axes]
    
    for idx, (pair_name, metrics) in enumerate(pair_metrics.items()):
        distances = metrics['distances']
        frames = np.arange(len(distances))
        
        # Smooth for trend visualization
        if len(distances) > 5:
            window_len = min(11, len(distances) if len(distances) % 2 == 1 else len(distances) - 1)
            smoothed = savgol_filter(distances, window_len, 3)
        else:
            smoothed = distances
        
        axes[idx].plot(frames, distances, alpha=0.5, label='Raw', color='blue', linewidth=1)
        axes[idx].plot(frames, smoothed, label='Smoothed', color='red', linewidth=2)
        axes[idx].axhline(y=metrics['mean'], color='green', linestyle='--', alpha=0.5, label='Mean')
        axes[idx].set_xlabel('Frame')
        axes[idx].set_ylabel('Distance (pixels)')
        axes[idx].set_title(f'{pair_name} - Range: {metrics["range"]:.2f}px, Smoothness: {metrics["normalized_smoothness"]:.4f}')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(ofu.OUTPUT_DIR, f"{video_name}_dynamic.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved: {plot_path}")
    plt.close()


# =========================================
# VISUALIZATION
# =========================================
def visualize_landmark_tracking(video_path, df, test_type='stability'):
    """Visualize landmark tracking on video."""
    if not ENABLE_VIDEO_VISUALIZATION:
        print(f"Skipping video visualization")
        return
    
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATION: {video_name} ({test_type.upper()})")
    print(f"SPACE=Pause, 'q'=Quit, '+/-'=Speed")
    print(f"{'='*60}")
    
    frame_idx, paused, delay_ms = 0, False, PLAYBACK_DELAY_MS
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]
    
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
                
                # Draw colored landmarks by group
                ofu.draw_landmarks(frame, xs, ys, show_points=True, show_box=False, color_by_group=True)
                
                # Draw measurement lines
                for pair_name, (idx1, idx2) in LANDMARK_PAIRS.items():
                    x1, y1 = int(xs[idx1]), int(ys[idx1])
                    x2, y2 = int(xs[idx2]), int(ys[idx2])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    dist = calculate_distance(xs[idx1], ys[idx1], xs[idx2], ys[idx2])
                    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.putText(frame, f"{dist:.1f}", (mid_x, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                status = f"Frame {frame_idx}: DETECTED"
                color = (0, 255, 0)
            else:
                status = f"Frame {frame_idx}: NOT DETECTED"
                color = (0, 0, 255)
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Test: {test_type.upper()}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            current_frame = frame.copy()
            frame_idx += 1
        else:
            frame = current_frame.copy()
            cv2.putText(frame, "PAUSED", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(f"Landmark Tracking - {video_name}", frame)
        
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


# =========================================
# MAIN EXECUTION
# =========================================
def generate_summary_report(all_results):
    """Generate final summary report."""
    if not all_results:
        return
    
    print(f"\n{'='*80}\nTEST B: SUMMARY REPORT\n{'='*80}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'-'*80}")
    
    stability_tests = [r for r in all_results if r['type'] == 'stability']
    if stability_tests:
        print(f"\nSTABILITY TESTS ({len(stability_tests)} videos):")
        print(f"{'-'*80}")
        for test in stability_tests:
            res = test['results']
            status = '✓ PASS' if res['passed'] else '✗ FAIL'
            print(f"{test['video']}: {status}")
            print(f"  Max Std Dev: {res['max_std']:.3f}px (threshold: {STABILITY_THRESHOLD})")
            if not res['passed']:
                print(f"  Failed Landmarks: {res['num_failed']}/{res['num_total']}")
    
    dynamic_tests = [r for r in all_results if r['type'] == 'dynamic']
    if dynamic_tests:
        print(f"\nDYNAMIC TESTS ({len(dynamic_tests)} videos):")
        print(f"{'-'*80}")
        for test in dynamic_tests:
            res = test['results']
            status = '✓ PASS' if res['passed'] else '✗ FAIL'
            print(f"{test['video']}: {status}")
            print(f"  Avg Smoothness: {res['avg_smoothness']:.4f}")
            for pair_name, smoothness in res['smoothness_scores'].items():
                print(f"    {pair_name}: {smoothness:.4f}")
    
    print(f"\n{'='*80}\n")


def main():
    """Main execution."""
    print(f"\n{'='*80}")
    print(f"TEST B: FACIAL LANDMARK TRACKING VALIDATION")
    print(f"{'='*80}")
    print(f"Purpose: Verify accurate landmark tracking during:")
    print(f"  1. Stability: Neutral expressions (≤2px variation)")
    print(f"  2. Dynamic: Exaggerated expressions (smooth tracking)\n")
    
    if not ofu.check_openface_binary():
        return
    
    # Find test videos
    stability_videos = ofu.find_videos(['*stability*.mp4', '*neutral*.mp4', '*still*.mp4'])
    dynamic_videos = ofu.find_videos(['*dynamic*.mp4', '*expression*.mp4', '*exertion*.mp4'])
    
    print(f"{'-'*80}")
    print(f"Stability videos ({len(stability_videos)}):")
    for vf in stability_videos:
        print(f"  - {os.path.basename(vf)}")
    print(f"\nDynamic videos ({len(dynamic_videos)}):")
    for vf in dynamic_videos:
        print(f"  - {os.path.basename(vf)}")
    print(f"{'-'*80}")
    
    if not stability_videos and not dynamic_videos:
        print(f"\nERROR: No test videos found!")
        print(f"Add videos to {ofu.VIDEOS_DIR} with naming:")
        print(f"  Stability: *stability*, *neutral*, *still*")
        print(f"  Dynamic: *dynamic*, *expression*, *exertion*")
        return
    
    if ENABLE_VIDEO_VISUALIZATION:
        print(f"\n📹 Video visualization enabled")
    else:
        print(f"\n⚡ Plots only (no video)")
    
    all_results = []
    
    # Stability tests
    for video_path in stability_videos:
        video_name = os.path.basename(video_path).split('.')[0]
        print(f"\n{'='*80}\nProcessing Stability: {video_name}\n{'='*80}")
        
        try:
            csv_path = ofu.run_openface(video_path)
            df = ofu.load_landmark_data(csv_path, success_only=True)
            results, stability_df = stability_test(df, video_name)
            plot_stability_heatmap(stability_df, video_name)
            visualize_landmark_tracking(video_path, df, test_type='stability')
            
            all_results.append({'type': 'stability', 'video': video_name, 'results': results})
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Dynamic tests
    for video_path in dynamic_videos:
        video_name = os.path.basename(video_path).split('.')[0]
        print(f"\n{'='*80}\nProcessing Dynamic: {video_name}\n{'='*80}")
        
        try:
            csv_path = ofu.run_openface(video_path)
            df = ofu.load_landmark_data(csv_path, success_only=True)
            results = dynamic_test(df, video_name)
            plot_dynamic_tracking(results, video_name)
            visualize_landmark_tracking(video_path, df, test_type='dynamic')
            
            all_results.append({'type': 'dynamic', 'video': video_name, 'results': results})
        except Exception as e:
            print(f"ERROR: {e}")
    
    generate_summary_report(all_results)


if __name__ == "__main__":
    main()
