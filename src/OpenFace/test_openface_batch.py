#!/usr/bin/env python3
"""
Run OpenFace extraction in small batches to manage WSL memory.
Processes videos in chunks, fully exiting Python between chunks to release memory.
"""
try:
    from .openface_feature_extractor import OpenFaceExtractor
    from .impute_missing_features import identify_missing_videos, impute_by_rpe_average, load_rpe_labels
    from . import openface_utils as ofu
except ImportError:
    from openface_feature_extractor import OpenFaceExtractor
    from impute_missing_features import identify_missing_videos, impute_by_rpe_average, load_rpe_labels
    import openface_utils as ofu
import pandas as pd
import glob
import os
import sys
import gc

BATCH_SIZE = 30  # Process 30 videos per batch, then exit to release memory

def get_processed_videos(output_dir):
    """Check which videos are already cached."""
    if not os.path.exists(output_dir):
        return set()
    
    csv_files = glob.glob(os.path.join(output_dir, "*_pose_guided.csv"))
    processed = set()
    for csv_file in csv_files:
        basename = os.path.basename(csv_file).replace("_pose_guided.csv", "")
        # Try to match original video name
        for ext in ['.mp4', '.mov', '.MOV', '.MP4']:
            processed.add(basename + ext)
    return processed

def process_batch(video_paths, start_idx, batch_size, output_csv):
    """Process a batch of videos."""
    end_idx = min(start_idx + batch_size, len(video_paths))
    batch = video_paths[start_idx:end_idx]
    
    print(f"\n{'='*80}")
    print(f"BATCH: Processing videos {start_idx+1} to {end_idx} (of {len(video_paths)} total)")
    print(f"{'='*80}\n")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    openface_cache_dir = os.path.join(project_root, 'output', 'openface')
    
    results = []
    for idx, video_path in enumerate(batch, start_idx+1):
        print(f"[{idx}/{len(video_paths)}] Processing: {os.path.basename(video_path)}")
        
        # Fresh extractor for each video
        extractor = OpenFaceExtractor(
            use_pose_guidance=True,
            max_only=True,
            load_minimal_columns=True,
            sample_fps=None,
            verbose=True
        )
        
        try:
            features = extractor.extract_from_video(video_path)
            result = {'video_name': os.path.basename(video_path)}
            result.update({k: v for k, v in features.items() if k != 'metadata'})
            results.append(result)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'video_name': os.path.basename(video_path), 'error': str(e)})
        
        finally:
            del extractor
            if (idx - start_idx) % 5 == 0:
                gc.collect()
    
    # Save batch results
    if results and output_csv:
        batch_df = pd.DataFrame(results)
        batch_path = output_csv.replace('.csv', f'_batch_{start_idx+1}_{end_idx}.csv')
        batch_df.to_csv(batch_path, index=False)
        print(f"\n[SAVED] Batch results: {os.path.basename(batch_path)}")
    
    return end_idx

def rebuild_from_cache(output_csv, cache_dir):
    """Rebuild the complete features CSV from all cached OpenFace outputs."""
    csv_files = glob.glob(os.path.join(cache_dir, "*_pose_guided.csv"))
    
    if not csv_files:
        print("No cached OpenFace outputs found.")
        return None
    
    print(f"\nRebuilding complete CSV from {len(csv_files)} cached videos...")
    results = []
    
    # Create extractor to process cached files
    extractor = OpenFaceExtractor(
        use_pose_guidance=False,  # Already processed
        max_only=True,
        load_minimal_columns=True,
        sample_fps=None,
        verbose=False
    )
    
    for idx, csv_file in enumerate(csv_files, 1):
        video_name = os.path.basename(csv_file).replace("_pose_guided.csv", ".mp4")
        
        if idx % 50 == 0 or idx == len(csv_files):
            print(f"  Processing {idx}/{len(csv_files)}...")
        
        try:
            # Extract features from cached CSV
            features = extractor.extract_from_cached_csv(csv_file, video_name)
            result = {'video_name': video_name}
            result.update({k: v for k, v in features.items() if k != 'metadata'})
            results.append(result)
            
        except Exception as e:
            print(f"  ERROR processing {video_name}: {e}")
            results.append({'video_name': video_name, 'error': str(e)})
    
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_csv, index=False)
        print(f"✓ Rebuilt CSV: {output_csv} ({len(final_df)} videos)")
        return final_df
    
    return None

def combine_batches(output_csv):
    """Combine all batch CSVs into final output."""
    pattern = output_csv.replace('.csv', '_batch_*.csv')
    batch_files = sorted(glob.glob(pattern))
    
    if not batch_files:
        print("No batch files found to combine.")
        return
    
    print(f"\nCombining {len(batch_files)} batch files...")
    all_dfs = []
    for batch_file in batch_files:
        df = pd.read_csv(batch_file)
        all_dfs.append(df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Final combined CSV: {output_csv} ({len(final_df)} videos)")
    
    return final_df

if __name__ == "__main__":
    video_folder = "../../lifting_videos/Augmented/"
    output_csv = "../../Train_Outputs/openface_features_all.csv"
    rpe_csv = "../../lifting_videos/Augmented/dataset_labelled.csv"
    
    # Get batch index from command line (if running multiple processes)
    if len(sys.argv) > 1:
        batch_idx = int(sys.argv[1])
    else:
        batch_idx = 0  # Process from beginning
    
    # Find all videos
    print(f"Scanning folder: {video_folder}")
    patterns = ['**/*.mp4', '**/*.mov', '**/*.MOV', '**/*.MP4']
    video_paths = []
    for pattern in patterns:
        video_paths.extend(glob.glob(os.path.join(video_folder, pattern), recursive=True))
    video_paths = sorted(list(set(video_paths)))
    print(f"Found {len(video_paths)} videos")
    
    # Filter out already processed
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = os.path.join(project_root, 'output', 'openface')
    processed = get_processed_videos(cache_dir)
    
    remaining = [v for v in video_paths if os.path.basename(v) not in processed]
    if remaining:
        print(f"Already processed: {len(video_paths) - len(remaining)}")
        print(f"Remaining: {len(remaining)}")
        video_paths = remaining
    else:
        print("All videos already processed! Rebuilding complete CSV from cache.")
        final_df = rebuild_from_cache(output_csv, cache_dir)
        
        # Impute missing if needed
        if final_df is not None and 'error' in final_df.columns:
            failed = identify_missing_videos(final_df)
            if len(failed) > 0:
                rpe_df = load_rpe_labels(rpe_csv)
                final_df, _ = impute_by_rpe_average(final_df, rpe_df, failed)
                final_df.to_csv(output_csv, index=False)
                print(f"Imputed {len(failed)} failed videos")
        
        print(f"\n✓ Complete! Final CSV: {output_csv}")
        sys.exit(0)
    
    # Process one batch
    start = batch_idx * BATCH_SIZE
    if start < len(video_paths):
        next_idx = process_batch(video_paths, start, BATCH_SIZE, output_csv)
        
        if next_idx < len(video_paths):
            print(f"\n{'='*80}")
            print(f"Batch complete. {len(video_paths) - next_idx} videos remaining.")
            print(f"Run again to process next batch:")
            print(f"  python test_openface_batch.py {batch_idx + 1}")
            print(f"{'='*80}\n")
            sys.exit(1)  # Non-zero to indicate more work
        else:
            print(f"\n{'='*80}")
            print("ALL VIDEOS PROCESSED!")
            print(f"{'='*80}\n")
    
    # Rebuild from ALL cached files (not just batch files)
    # This ensures we capture everything even if batches were run at different times
    final_df = rebuild_from_cache(output_csv, cache_dir)
    
    # Impute missing
    if final_df is not None and 'error' in final_df.columns:
        failed = identify_missing_videos(final_df)
        if len(failed) > 0:
            rpe_df = load_rpe_labels(rpe_csv)
            final_df, _ = impute_by_rpe_average(final_df, rpe_df, failed)
            final_df.to_csv(output_csv, index=False)
            print(f"Imputed {len(failed)} failed videos")
    
    print(f"\n✓ Complete! Final CSV: {output_csv}")
