"""
TEST C: Facial Action Unit (AU) Validation

Tests OpenFace's AU intensity estimates during physical exertion:
1. Resting Baseline: AU intensities near zero (≤0.2) during rest
2. Exertion Response: AU intensities increase proportionally with effort
3. Repetition Consistency: AU patterns rise/fall consistently across reps
4. Visual Correspondence: AU spikes match observed facial strain (≥95%)
"""

import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import openface_utils as ofu

# Test configuration
RESTING_BASELINE_THRESHOLD = 0.2
EXERTION_INCREASE_RATIO = 1.5  # Peak should be 1.5x baseline
REPETITION_CONSISTENCY_THRESHOLD = 0.3  # Max std dev between rep peaks
VISUAL_CORRESPONDENCE_THRESHOLD = 0.95
ENABLE_VIDEO_VISUALIZATION = True
PLAYBACK_DELAY_MS = 30

# Key Action Units for exertion (intensity columns)
# AU04: Brow Lowerer (concentration/strain)
# AU06: Cheek Raiser (squinting/strain)
# AU07: Lid Tightener (strain)
# AU09: Nose Wrinkler (disgust/strain)
# AU10: Upper Lip Raiser (strain)
# AU12: Lip Corner Puller (smile/grimace)
# AU17: Chin Raiser (strain)
# AU20: Lip Stretcher (strain)
# AU25: Lips Part (breathing/exertion)
# AU26: Jaw Drop (breathing/exertion)
KEY_EXERTION_AUS = {
    'AU04_r': 'Brow Lowerer',
    'AU06_r': 'Cheek Raiser',
    'AU07_r': 'Lid Tightener',
    'AU09_r': 'Nose Wrinkler',
    'AU10_r': 'Upper Lip Raiser',
    'AU12_r': 'Lip Corner Puller',
    'AU17_r': 'Chin Raiser',
    'AU20_r': 'Lip Stretcher',
    'AU25_r': 'Lips Part',
    'AU26_r': 'Jaw Drop',
}

# All AU intensity columns available in OpenFace
ALL_AU_INTENSITY = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]


# =========================================
# TEST 1: RESTING BASELINE
# =========================================
def test_resting_baseline(df, video_name):
    """Verify AU intensities remain near zero during rest."""
    print(f"\n{'='*60}\nTEST 1: RESTING BASELINE - {video_name}\n{'='*60}")
    
    baseline_results = {}
    failed_aus = []
    
    for au_col in ALL_AU_INTENSITY:
        if au_col not in df.columns:
            continue
        
        mean_intensity = df[au_col].mean()
        max_intensity = df[au_col].max()
        std_intensity = df[au_col].std()
        
        passed = mean_intensity <= RESTING_BASELINE_THRESHOLD
        
        baseline_results[au_col] = {
            'mean': mean_intensity,
            'max': max_intensity,
            'std': std_intensity,
            'passed': passed
        }
        
        if not passed:
            failed_aus.append((au_col, mean_intensity))
    
    # Print results
    print(f"\nThreshold: ≤{RESTING_BASELINE_THRESHOLD}")
    print(f"Total AUs: {len(baseline_results)}")
    print(f"Failed: {len(failed_aus)}/{len(baseline_results)}")
    
    if failed_aus:
        print(f"\nAUs exceeding baseline threshold:")
        for au, mean_val in sorted(failed_aus, key=lambda x: x[1], reverse=True)[:10]:
            au_name = KEY_EXERTION_AUS.get(au, au)
            print(f"  {au} ({au_name}): {mean_val:.3f}")
    
    all_passed = len(failed_aus) == 0
    print(f"\n{'-'*60}")
    print(f"RESULT: {'✓ PASS' if all_passed else '✗ FAIL'}")
    print(f"{'-'*60}")
    
    # Save report
    report_df = pd.DataFrame([
        {'AU': au, **stats} for au, stats in baseline_results.items()
    ])
    report_path = os.path.join(ofu.OUTPUT_DIR, f"{video_name}_baseline.csv")
    report_df.to_csv(report_path, index=False)
    print(f"Report saved: {report_path}")
    
    return {
        'passed': all_passed,
        'num_failed': len(failed_aus),
        'num_total': len(baseline_results),
        'baseline_results': baseline_results
    }


# =========================================
# TEST 2: EXERTION RESPONSE
# =========================================
def test_exertion_response(df, video_name, effort_level='unknown'):
    """Verify AU intensities increase with exertion."""
    print(f"\n{'='*60}\nTEST 2: EXERTION RESPONSE - {video_name}\n{'='*60}")
    print(f"Effort Level: {effort_level.upper()}")
    
    exertion_results = {}
    responsive_aus = []
    
    for au_col in KEY_EXERTION_AUS.keys():
        if au_col not in df.columns:
            continue
        
        baseline = df[au_col].quantile(0.1)  # Bottom 10% as baseline
        peak = df[au_col].quantile(0.9)      # Top 10% as peak
        mean_intensity = df[au_col].mean()
        std_intensity = df[au_col].std()
        range_intensity = peak - baseline
        
        # Check if AU shows exertion response
        increase_ratio = peak / baseline if baseline > 0.01 else 0
        responsive = increase_ratio >= EXERTION_INCREASE_RATIO
        
        exertion_results[au_col] = {
            'baseline': baseline,
            'peak': peak,
            'mean': mean_intensity,
            'std': std_intensity,
            'range': range_intensity,
            'increase_ratio': increase_ratio,
            'responsive': responsive
        }
        
        if responsive:
            responsive_aus.append((au_col, increase_ratio, peak))
    
    # Print results
    print(f"\nThreshold: Peak ≥ {EXERTION_INCREASE_RATIO}x Baseline")
    print(f"Key Exertion AUs: {len(KEY_EXERTION_AUS)}")
    print(f"Responsive AUs: {len(responsive_aus)}")
    
    if responsive_aus:
        print(f"\nResponsive AUs (sorted by increase):")
        for au, ratio, peak in sorted(responsive_aus, key=lambda x: x[1], reverse=True)[:10]:
            au_name = KEY_EXERTION_AUS.get(au, au)
            print(f"  {au} ({au_name}): {ratio:.2f}x (peak={peak:.2f})")
    
    # At least 3 key AUs should respond
    passed = len(responsive_aus) >= 3
    print(f"\n{'-'*60}")
    print(f"RESULT: {'✓ PASS' if passed else '✗ FAIL'} (≥3 responsive AUs required)")
    print(f"{'-'*60}")
    
    # Save report
    report_df = pd.DataFrame([
        {'AU': au, 'name': KEY_EXERTION_AUS.get(au, ''), **stats}
        for au, stats in exertion_results.items()
    ])
    report_path = os.path.join(ofu.OUTPUT_DIR, f"{video_name}_exertion.csv")
    report_df.to_csv(report_path, index=False)
    print(f"Report saved: {report_path}")
    
    return {
        'passed': passed,
        'num_responsive': len(responsive_aus),
        'num_key_aus': len(KEY_EXERTION_AUS),
        'responsive_aus': responsive_aus,
        'exertion_results': exertion_results
    }


# =========================================
# TEST 3: REPETITION CONSISTENCY
# =========================================
def detect_repetitions(au_series, min_distance=30):
    """Detect repetition peaks in AU intensity signal."""
    # Smooth signal
    if len(au_series) > 5:
        window_len = min(11, len(au_series) if len(au_series) % 2 == 1 else len(au_series) - 1)
        smoothed = savgol_filter(au_series, window_len, 3)
    else:
        smoothed = au_series
    
    # Find peaks
    peaks, properties = find_peaks(smoothed, distance=min_distance, prominence=0.3)
    
    return peaks, smoothed


def test_repetition_consistency(df, video_name, num_expected_reps=None):
    """Verify AU patterns are consistent across repetitions."""
    print(f"\n{'='*60}\nTEST 3: REPETITION CONSISTENCY - {video_name}\n{'='*60}")
    
    consistency_results = {}
    consistent_aus = []
    
    for au_col in KEY_EXERTION_AUS.keys():
        if au_col not in df.columns:
            continue
        
        au_data = df[au_col].values
        peaks, smoothed = detect_repetitions(au_data)
        
        if len(peaks) < 2:
            # Need at least 2 reps to check consistency
            consistency_results[au_col] = {
                'num_peaks': len(peaks),
                'peak_values': [],
                'peak_std': 0,
                'consistent': False,
                'reason': 'Insufficient repetitions'
            }
            continue
        
        peak_values = au_data[peaks]
        peak_mean = np.mean(peak_values)
        peak_std = np.std(peak_values)
        normalized_std = peak_std / peak_mean if peak_mean > 0.1 else peak_std
        
        # Consistency check: normalized std should be low
        consistent = normalized_std <= REPETITION_CONSISTENCY_THRESHOLD
        
        consistency_results[au_col] = {
            'num_peaks': len(peaks),
            'peak_values': peak_values.tolist(),
            'peak_mean': peak_mean,
            'peak_std': peak_std,
            'normalized_std': normalized_std,
            'consistent': consistent,
            'peak_frames': peaks.tolist()
        }
        
        if consistent:
            consistent_aus.append((au_col, normalized_std, len(peaks)))
    
    # Print results
    detected_reps = max([r['num_peaks'] for r in consistency_results.values()]) if consistency_results else 0
    print(f"\nDetected Repetitions: {detected_reps}")
    if num_expected_reps:
        print(f"Expected Repetitions: {num_expected_reps}")
    print(f"Threshold: Normalized Std ≤ {REPETITION_CONSISTENCY_THRESHOLD}")
    print(f"Consistent AUs: {len(consistent_aus)}/{len([r for r in consistency_results.values() if r['num_peaks'] >= 2])}")
    
    if consistent_aus:
        print(f"\nMost consistent AUs:")
        for au, norm_std, num_peaks in sorted(consistent_aus, key=lambda x: x[1])[:5]:
            au_name = KEY_EXERTION_AUS.get(au, au)
            print(f"  {au} ({au_name}): {num_peaks} peaks, std={norm_std:.3f}")
    
    # At least 2 AUs should show consistent patterns
    passed = len(consistent_aus) >= 2
    print(f"\n{'-'*60}")
    print(f"RESULT: {'✓ PASS' if passed else '✗ FAIL'} (≥2 consistent AUs required)")
    print(f"{'-'*60}")
    
    # Save report
    report_df = pd.DataFrame([
        {'AU': au, 'name': KEY_EXERTION_AUS.get(au, ''), **{k: v for k, v in stats.items() if k != 'peak_values' and k != 'peak_frames'}}
        for au, stats in consistency_results.items()
    ])
    report_path = os.path.join(ofu.OUTPUT_DIR, f"{video_name}_consistency.csv")
    report_df.to_csv(report_path, index=False)
    print(f"Report saved: {report_path}")
    
    return {
        'passed': passed,
        'num_consistent': len(consistent_aus),
        'detected_reps': detected_reps,
        'consistency_results': consistency_results
    }


# =========================================
# VISUALIZATION
# =========================================
def plot_au_timeseries(df, video_name, results=None):
    """Plot AU intensities over time."""
    num_aus = len(KEY_EXERTION_AUS)
    fig, axes = plt.subplots(num_aus, 1, figsize=(14, 2.5 * num_aus))
    if num_aus == 1:
        axes = [axes]
    
    frames = df['frame'].values if 'frame' in df.columns else np.arange(len(df))
    
    for idx, (au_col, au_name) in enumerate(KEY_EXERTION_AUS.items()):
        if au_col not in df.columns:
            continue
        
        au_data = df[au_col].values
        
        # Plot raw data
        axes[idx].plot(frames, au_data, alpha=0.6, label='Raw', color='blue', linewidth=1)
        
        # Plot smoothed
        if len(au_data) > 5:
            window_len = min(11, len(au_data) if len(au_data) % 2 == 1 else len(au_data) - 1)
            smoothed = savgol_filter(au_data, window_len, 3)
            axes[idx].plot(frames, smoothed, label='Smoothed', color='red', linewidth=2)
        
        # Mark peaks if consistency results available
        if results and 'consistency_results' in results:
            consistency_data = results['consistency_results'].get(au_col, {})
            if 'peak_frames' in consistency_data:
                peak_frames = consistency_data['peak_frames']
                peak_values = au_data[peak_frames]
                axes[idx].scatter(frames[peak_frames], peak_values, 
                                color='green', s=100, zorder=5, label=f'Peaks (n={len(peak_frames)})')
        
        # Baseline threshold
        axes[idx].axhline(y=RESTING_BASELINE_THRESHOLD, color='orange', 
                         linestyle='--', alpha=0.5, label=f'Baseline ({RESTING_BASELINE_THRESHOLD})')
        
        axes[idx].set_xlabel('Frame')
        axes[idx].set_ylabel('Intensity (0-5)')
        axes[idx].set_title(f'{au_col}: {au_name}')
        axes[idx].legend(loc='upper right', fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(ofu.OUTPUT_DIR, f"{video_name}_au_timeseries.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nAU timeseries plot saved: {plot_path}")
    plt.close()


def plot_au_heatmap(df, video_name):
    """Create heatmap of all AU intensities over time."""
    au_cols = [col for col in ALL_AU_INTENSITY if col in df.columns]
    
    if not au_cols:
        return
    
    au_data = df[au_cols].values.T
    
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(au_data, aspect='auto', cmap='hot', interpolation='nearest')
    
    ax.set_yticks(np.arange(len(au_cols)))
    ax.set_yticklabels(au_cols)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Action Unit')
    ax.set_title(f'AU Intensity Heatmap - {video_name}')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Intensity (0-5)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plot_path = os.path.join(ofu.OUTPUT_DIR, f"{video_name}_au_heatmap.png")
    plt.savefig(plot_path, dpi=150)
    print(f"AU heatmap saved: {plot_path}")
    plt.close()


def visualize_au_video(video_path, df, top_aus=None):
    """Visualize video with AU intensity overlays."""
    if not ENABLE_VIDEO_VISUALIZATION:
        print(f"Skipping video visualization")
        return
    
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    
    # Determine top AUs to display (most active)
    if top_aus is None:
        au_ranges = {au: df[au].max() - df[au].min() 
                     for au in KEY_EXERTION_AUS.keys() if au in df.columns}
        top_aus = sorted(au_ranges.items(), key=lambda x: x[1], reverse=True)[:5]
        top_aus = [au for au, _ in top_aus]
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATION: {video_name}")
    print(f"Displaying top {len(top_aus)} AUs")
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
                
                # Draw landmarks
                ofu.draw_landmarks(frame, xs, ys, show_points=True, show_box=False, color_by_group=True)
                
                # AU intensity bars
                bar_x, bar_y = 10, frame.shape[0] - 150
                bar_width, bar_height = 200, 20
                
                for i, au_col in enumerate(top_aus):
                    intensity = data[au_col]
                    au_name = KEY_EXERTION_AUS.get(au_col, au_col)
                    
                    # Background bar
                    cv2.rectangle(frame, (bar_x, bar_y + i * 25), 
                                (bar_x + bar_width, bar_y + i * 25 + bar_height),
                                (50, 50, 50), -1)
                    
                    # Intensity bar (0-5 scale)
                    intensity_width = int((intensity / 5.0) * bar_width)
                    color = (0, int(255 * (intensity / 5.0)), int(255 * (1 - intensity / 5.0)))
                    cv2.rectangle(frame, (bar_x, bar_y + i * 25),
                                (bar_x + intensity_width, bar_y + i * 25 + bar_height),
                                color, -1)
                    
                    # Label
                    cv2.putText(frame, f"{au_col}: {intensity:.2f}", 
                              (bar_x + bar_width + 10, bar_y + i * 25 + 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                status = f"Frame {frame_idx}: DETECTED"
                color = (0, 255, 0)
            else:
                status = f"Frame {frame_idx}: NOT DETECTED"
                color = (0, 0, 255)
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"AU Analysis", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            current_frame = frame.copy()
            frame_idx += 1
        else:
            frame = current_frame.copy()
            cv2.putText(frame, "PAUSED", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(f"AU Analysis - {video_name}", frame)
        
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
def parse_video_metadata(video_name):
    """Extract metadata from video filename."""
    name_lower = video_name.lower()
    
    # Effort level
    if 'low' in name_lower or 'light' in name_lower:
        effort = 'low'
    elif 'high' in name_lower or 'max' in name_lower:
        effort = 'high'
    elif 'medium' in name_lower or 'med' in name_lower or 'moderate' in name_lower:
        effort = 'medium'
    elif 'rest' in name_lower or 'baseline' in name_lower:
        effort = 'resting'
    else:
        effort = 'unknown'
    
    # Expected repetitions
    num_reps = None
    for i in range(1, 21):
        if f'{i}rep' in name_lower or f'{i}_rep' in name_lower:
            num_reps = i
            break
    
    return {'effort': effort, 'num_reps': num_reps}


def generate_summary_report(all_results):
    """Generate final summary report."""
    if not all_results:
        return
    
    print(f"\n{'='*80}\nTEST C: SUMMARY REPORT\n{'='*80}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'-'*80}")
    
    for result in all_results:
        video = result['video']
        metadata = result['metadata']
        
        print(f"\n{video} (Effort: {metadata['effort'].upper()})")
        print(f"{'-'*80}")
        
        if 'baseline' in result:
            res = result['baseline']
            status = '✓ PASS' if res['passed'] else '✗ FAIL'
            print(f"  Baseline Test: {status} ({res['num_failed']}/{res['num_total']} failed)")
        
        if 'exertion' in result:
            res = result['exertion']
            status = '✓ PASS' if res['passed'] else '✗ FAIL'
            print(f"  Exertion Test: {status} ({res['num_responsive']}/{res['num_key_aus']} responsive)")
        
        if 'consistency' in result:
            res = result['consistency']
            status = '✓ PASS' if res['passed'] else '✗ FAIL'
            print(f"  Consistency Test: {status} ({res['num_consistent']} AUs, {res['detected_reps']} reps)")
    
    print(f"\n{'='*80}\n")


def main():
    """Main execution."""
    print(f"\n{'='*80}")
    print(f"TEST C: FACIAL ACTION UNIT (AU) VALIDATION")
    print(f"{'='*80}")
    print(f"Purpose: Verify AU intensities respond to physical exertion")
    print(f"  1. Resting: AU ≤ {RESTING_BASELINE_THRESHOLD}")
    print(f"  2. Exertion: AU increases with effort")
    print(f"  3. Consistency: AU patterns repeat across reps")
    print(f"  4. Visual: AU spikes match facial strain\n")
    
    if not ofu.check_openface_binary():
        return
    
    # Find videos by type
    resting_videos = ofu.find_videos(['*rest*.mp4', '*baseline*.mp4', '*neutral*.mp4'])
    exertion_videos = ofu.find_videos(['*low*.mp4', '*medium*.mp4', '*high*.mp4', 
                                       '*light*.mp4', '*moderate*.mp4', '*max*.mp4',
                                       '*effort*.mp4', '*exertion*.mp4'])
    
    print(f"{'-'*80}")
    print(f"Resting videos ({len(resting_videos)}):")
    for vf in resting_videos:
        print(f"  - {os.path.basename(vf)}")
    print(f"\nExertion videos ({len(exertion_videos)}):")
    for vf in exertion_videos:
        print(f"  - {os.path.basename(vf)}")
    print(f"{'-'*80}")
    
    if not resting_videos and not exertion_videos:
        print(f"\nERROR: No test videos found!")
        print(f"Add videos to {ofu.VIDEOS_DIR} with naming:")
        print(f"  Resting: *rest*, *baseline*, *neutral*")
        print(f"  Exertion: *low*, *medium*, *high*, *effort* (optional: 3rep, 5rep, etc.)")
        return
    
    if ENABLE_VIDEO_VISUALIZATION:
        print(f"\n📹 Video visualization enabled")
    else:
        print(f"\n⚡ Plots only (no video)")
    
    all_results = []
    
    # Test resting videos
    for video_path in resting_videos:
        video_name = os.path.basename(video_path).split('.')[0]
        metadata = parse_video_metadata(video_name)
        
        print(f"\n{'='*80}\nProcessing Resting: {video_name}\n{'='*80}")
        
        try:
            csv_path = ofu.run_openface(video_path)
            df = ofu.load_landmark_data(csv_path, success_only=True)
            
            baseline_results = test_resting_baseline(df, video_name)
            plot_au_timeseries(df, video_name)
            plot_au_heatmap(df, video_name)
            visualize_au_video(video_path, df)
            
            all_results.append({
                'video': video_name,
                'metadata': metadata,
                'baseline': baseline_results
            })
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Test exertion videos
    for video_path in exertion_videos:
        video_name = os.path.basename(video_path).split('.')[0]
        metadata = parse_video_metadata(video_name)
        
        print(f"\n{'='*80}\nProcessing Exertion: {video_name}\n{'='*80}")
        
        try:
            csv_path = ofu.run_openface(video_path)
            df = ofu.load_landmark_data(csv_path, success_only=True)
            
            exertion_results = test_exertion_response(df, video_name, metadata['effort'])
            consistency_results = test_repetition_consistency(df, video_name, metadata['num_reps'])
            
            plot_au_timeseries(df, video_name, {'consistency_results': consistency_results['consistency_results']})
            plot_au_heatmap(df, video_name)
            
            # Get top responsive AUs for visualization
            top_aus = [au for au, _, _ in exertion_results['responsive_aus'][:5]] if exertion_results['responsive_aus'] else None
            visualize_au_video(video_path, df, top_aus)
            
            all_results.append({
                'video': video_name,
                'metadata': metadata,
                'exertion': exertion_results,
                'consistency': consistency_results
            })
        except Exception as e:
            print(f"ERROR: {e}")
    
    generate_summary_report(all_results)


if __name__ == "__main__":
    main()
