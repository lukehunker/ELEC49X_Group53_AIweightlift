#!/usr/bin/env python3
"""
Aggregate D-metric results from preprocess_*.py output files into master Excel spreadsheet.
Similar pattern to barspeed_to_excel.py - processes text output files and creates/updates
a single master dmetrics.xlsx file.
"""

import os
import re
import pandas as pd
from pathlib import Path

# Configuration
MMPOSE_DIR = "."  # Run from src/MMPose/
OUTPUT_DIR = "../Train_Outputs"
MASTER_EXCEL_PATH = os.path.join(OUTPUT_DIR, "dmetrics.xlsx")

# Exercise output file patterns
EXERCISE_PATTERNS = {
    "squat": "squat_outputs.txt",
    "bench_press": "bench_outputs.txt",
    "deadlift": "deadlift_outputs.txt"
}

def parse_output_file(filepath, exercise_type):
    """
    Parse a *_outputs.txt file to extract D-metric results.
    Format expected:
        Video_Name RESULT
        Video_Name First rep: (start, end)
        Video_Name Last rep : (start, end)
        Video_Name D value  : 1.234567
    """
    results = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Group lines by video (every 4 lines form one record)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for RESULT line
        if "RESULT" in line:
            # Extract video name (everything before " RESULT")
            video_name = line.split(" RESULT")[0].strip()
            
            # Parse next 3 lines  
            if i + 3 < len(lines):
                first_rep_line = lines[i + 1].strip()
                last_rep_line = lines[i + 2].strip()
                d_value_line = lines[i + 3].strip()
                
                # Extract frame numbers from "First rep: (123, 456)"
                first_match = re.search(r'First rep:\s*\((\d+),\s*(\d+)\)', first_rep_line)
                last_match = re.search(r'Last rep\s*:\s*\((\d+),\s*(\d+)\)', last_rep_line)
                d_match = re.search(r'D value\s*:\s*([\d.]+)', d_value_line)
                
                if first_match and last_match and d_match:
                    results.append({
                        'video_name': video_name,
                        'exercise': exercise_type,
                        'first_rep_start': int(first_match.group(1)),
                        'first_rep_end': int(first_match.group(2)),
                        'last_rep_start': int(last_match.group(1)),
                        'last_rep_end': int(last_match.group(2)),
                        'd_value': float(d_match.group(1))
                    })
                
                i += 4  # Move to next record
                continue
        
        i += 1
    
    return results

def main():
    print("\n==========================================")
    print("  D-METRIC AGGREGATION TO EXCEL")
    print("==========================================\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = []
    
    # Process each exercise type
    for exercise, output_file in EXERCISE_PATTERNS.items():
        filepath = os.path.join(MMPOSE_DIR, output_file)
        
        if not os.path.exists(filepath):
            print(f"[SKIP] {output_file} not found")
            continue
        
        print(f"[PROCESSING] {output_file}...")
        results = parse_output_file(filepath, exercise)
        
        if results:
            all_results.extend(results)
            print(f"  ✓ Found {len(results)} results for {exercise}")
        else:
            print(f"  ⚠ No results found in {output_file}")
    
    if not all_results:
        print("\n❌ No results found. Run extraction scripts first!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by exercise, then video name
    df = df.sort_values(['exercise', 'video_name']).reset_index(drop=True)
    
    # Save to Excel
    try:
        df.to_excel(MASTER_EXCEL_PATH, index=False, sheet_name='D-Metrics')
        print(f"\n✓ Saved {len(df)} results to: {MASTER_EXCEL_PATH}")
    except PermissionError:
        print(f"\n❌ Could not save to {MASTER_EXCEL_PATH}")
        print("   Please close the Excel file if it is open!")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal videos processed: {len(df)}")
    print(f"\nBy exercise:")
    for exercise in df['exercise'].unique():
        count = len(df[df['exercise'] == exercise])
        avg_d = df[df['exercise'] == exercise]['d_value'].mean()
        print(f"  {exercise:12s}: {count:3d} videos, avg D = {avg_d:.4f}")
    
    print(f"\nD-value statistics (all exercises):")
    print(f"  Min:  {df['d_value'].min():.4f}")
    print(f"  Max:  {df['d_value'].max():.4f}")
    print(f"  Mean: {df['d_value'].mean():.4f}")
    print(f"  Std:  {df['d_value'].std():.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
