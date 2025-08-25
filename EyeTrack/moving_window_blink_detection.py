#!/usr/bin/env python3
"""
Moving Window Blink Rate Detection
Detects sudden changes in blink rate using a sliding time window.
"""

import csv
import numpy as np
import pandas as pd
from collections import deque

def create_sample_csv():
    """Create sample data with varying blink patterns"""
    
    # Sample data with different blink patterns over time
    sample_data = [
        # timestamp, left_eye_openness, right_eye_openness, left_pupil_diam, right_pupil_diam
        # Period 1: Normal blinking (0-3 seconds)
        [0.0, 0.8, 0.9, 3.2, 3.1],  # Normal
        [0.3, 0.1, 0.1, 3.1, 3.0],  # BLINK
        [0.6, 0.8, 0.9, 3.3, 3.2],  # Normal
        [0.9, 0.9, 0.8, 3.4, 3.3],  # Normal
        [1.2, 0.1, 0.1, 3.2, 3.1],  # BLINK
        [1.5, 0.8, 0.9, 3.5, 3.4],  # Normal
        [1.8, 0.9, 0.8, 3.6, 3.5],  # Normal
        [2.1, 0.1, 0.1, 3.3, 3.2],  # BLINK
        [2.4, 0.8, 0.9, 3.7, 3.6],  # Normal
        [2.7, 0.9, 0.8, 3.8, 3.7],  # Normal
        
        # Period 2: High stress - rapid blinking (3-6 seconds)
        [3.0, 0.1, 0.1, 3.9, 3.8],  # BLINK
        [3.2, 0.8, 0.9, 4.0, 3.9],  # Normal
        [3.4, 0.1, 0.1, 4.1, 4.0],  # BLINK
        [3.6, 0.8, 0.9, 4.2, 4.1],  # Normal
        [3.8, 0.1, 0.1, 4.3, 4.2],  # BLINK
        [4.0, 0.8, 0.9, 4.4, 4.3],  # Normal
        [4.2, 0.1, 0.1, 4.5, 4.4],  # BLINK
        [4.4, 0.8, 0.9, 4.6, 4.5],  # Normal
        [4.6, 0.1, 0.1, 4.7, 4.6],  # BLINK
        [4.8, 0.8, 0.9, 4.8, 4.7],  # Normal
        [5.0, 0.1, 0.1, 4.9, 4.8],  # BLINK
        [5.2, 0.8, 0.9, 5.0, 4.9],  # Normal
        [5.4, 0.1, 0.1, 5.1, 5.0],  # BLINK
        [5.6, 0.8, 0.9, 5.2, 5.1],  # Normal
        [5.8, 0.1, 0.1, 5.3, 5.2],  # BLINK
        [6.0, 0.8, 0.9, 5.4, 5.3],  # Normal
        
        # Period 3: Return to normal (6-9 seconds)
        [6.3, 0.1, 0.1, 5.5, 5.4],  # BLINK
        [6.6, 0.8, 0.9, 5.6, 5.5],  # Normal
        [6.9, 0.9, 0.8, 5.7, 5.6],  # Normal
        [7.2, 0.1, 0.1, 5.8, 5.7],  # BLINK
        [7.5, 0.8, 0.9, 5.9, 5.8],  # Normal
        [7.8, 0.9, 0.8, 6.0, 5.9],  # Normal
        [8.1, 0.1, 0.1, 6.1, 6.0],  # BLINK
        [8.4, 0.8, 0.9, 6.2, 6.1],  # Normal
        [8.7, 0.9, 0.8, 6.3, 6.2],  # Normal
        [9.0, 0.1, 0.1, 6.4, 6.3],  # BLINK
        [9.3, 0.8, 0.9, 6.5, 6.4],  # Normal
        
        # Period 4: Moderate stress (9-12 seconds)
        [9.6, 0.1, 0.1, 6.6, 6.5],  # BLINK
        [9.8, 0.8, 0.9, 6.7, 6.6],  # Normal
        [10.0, 0.1, 0.1, 6.8, 6.7],  # BLINK
        [10.2, 0.8, 0.9, 6.9, 6.8],  # Normal
        [10.4, 0.1, 0.1, 7.0, 6.9],  # BLINK
        [10.6, 0.8, 0.9, 7.1, 7.0],  # Normal
        [10.8, 0.1, 0.1, 7.2, 7.1],  # BLINK
        [11.0, 0.8, 0.9, 7.3, 7.2],  # Normal
        [11.2, 0.1, 0.1, 7.4, 7.3],  # BLINK
        [11.4, 0.8, 0.9, 7.5, 7.4],  # Normal
        [11.6, 0.1, 0.1, 7.6, 7.5],  # BLINK
        [11.8, 0.8, 0.9, 7.7, 7.6],  # Normal
        [12.0, 0.1, 0.1, 7.8, 7.7],  # BLINK
        [12.2, 0.8, 0.9, 7.9, 7.8],  # Normal
        
        # Period 5: Final normal period (12-15 seconds)
        [12.5, 0.1, 0.1, 8.0, 7.9],  # BLINK
        [12.8, 0.8, 0.9, 8.1, 8.0],  # Normal
        [13.1, 0.9, 0.8, 8.2, 8.1],  # Normal
        [13.4, 0.1, 0.1, 8.3, 8.2],  # BLINK
        [13.7, 0.8, 0.9, 8.4, 8.3],  # Normal
        [14.0, 0.9, 0.8, 8.5, 8.4],  # Normal
        [14.3, 0.1, 0.1, 8.6, 8.5],  # BLINK
        [14.6, 0.8, 0.9, 8.7, 8.6],  # Normal
        [14.9, 0.9, 0.8, 8.8, 8.7],  # Normal
        [15.2, 0.1, 0.1, 8.9, 8.8],  # BLINK
        [15.5, 0.8, 0.9, 9.0, 8.9],  # Normal
    ]
    
    csv_filename = "moving_window_demo.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'left_eye_openness', 'right_eye_openness', 
                        'left_pupil_diam', 'right_pupil_diam'])
        for row in sample_data:
            writer.writerow(row)
    
    print(f"Created sample CSV: {csv_filename}")
    print("Data pattern:")
    print("  0-3s: Normal blinking (3 blinks)")
    print("  3-6s: High stress - rapid blinking (8 blinks)")
    print("  6-9s: Return to normal (4 blinks)")
    print("  9-12s: Moderate stress (6 blinks)")
    print("  12-15s: Final normal period (4 blinks)")
    print()
    
    return csv_filename

def detect_blink(timestamp, left_openness, right_openness):
    blink_threshold = 0.3
    return left_openness < blink_threshold and right_openness < blink_threshold

def calculate_moving_window_blink_rate(df, window_size=2.0):
    
    results = []
    blink_threshold = 0.3
    refractory_period = 0.1  # Minimum time between blinks
    
    for i, row in df.iterrows():
        current_time = row['timestamp']
        window_start = current_time - window_size
        
        # Only analyze windows that start at or after the first data point
        if window_start < 0:
            results.append({
                'timestamp': current_time,
                'blink_rate': 0.0,
                'window_start': window_start,
                'window_end': current_time,
                'blinks_in_window': 0,
                'window_duration': 0.0
            })
            continue
        
        # Get all data points within the current window
        window_data = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= current_time)]
        
        if len(window_data) < 2:  # Need at least 2 points for meaningful calculation
            results.append({
                'timestamp': current_time,
                'blink_rate': 0.0,
                'window_start': window_start,
                'window_end': current_time,
                'blinks_in_window': 0,
                'window_duration': 0.0
            })
            continue
        
        # Detect blinks within the window
        blinks_in_window = []
        last_blink_time = -1
        
        for _, window_row in window_data.iterrows():
            left_open = window_row['left_eye_openness']
            right_open = window_row['right_eye_openness']
            blink_time = window_row['timestamp']
            
            # Check if this is a blink
            if left_open < blink_threshold and right_open < blink_threshold:

                if blink_time - last_blink_time > refractory_period:
                    blinks_in_window.append(blink_time)
                    last_blink_time = blink_time
        
        # Calculate blink rate for this window
        window_duration = current_time - window_start
        if window_duration > 0:
            blink_rate = len(blinks_in_window) / window_duration
        else:
            blink_rate = 0.0
        
        results.append({
            'timestamp': current_time,
            'blink_rate': blink_rate,
            'window_start': window_start,
            'window_end': current_time,
            'blinks_in_window': len(blinks_in_window),
            'window_duration': window_duration
        })
    
    return pd.DataFrame(results)


def detect_blink_rate_increases(blink_rates_df, threshold_increase=0.1):
    """
    Detect when blink rate increases significantly between consecutive windows
    
    Args:
        blink_rates_df: DataFrame with blink_rate over time
        threshold_increase: Minimum increase in blinks/second to trigger detection
    
    Returns:
        List of detected blink rate increases
    """
    
    increases = []
    previous_blink_rate = None
    
    for i, row in blink_rates_df.iterrows():
        current_blink_rate = row['blink_rate']
        timestamp = row['timestamp']
        
        # Skip first window only
        if i == 0:
            previous_blink_rate = current_blink_rate
            continue
        
        # Check for significant increase from previous window
        if previous_blink_rate is not None and current_blink_rate > previous_blink_rate + threshold_increase:
            increases.append({
                'timestamp': timestamp,
                'current_rate': current_blink_rate,
                'previous_rate': previous_blink_rate,
                'increase': current_blink_rate - previous_blink_rate,
                'window_start': row['window_start'],
                'window_end': row['window_end']
            })
        
        previous_blink_rate = current_blink_rate
    
    return increases

def analyze_moving_window_blinks(csv_filename):
    
    print(f"Analyzing moving window blink rates from: {csv_filename}")
    print("=" * 60)
    
    # Read CSV
    df = pd.read_csv(csv_filename)
    
    print("Sample data (first 10 rows):")
    print(df.head(10).to_string(index=False))
    print()
    
    # Calculate moving window blink rates
    window_size = 2.0  # 2-second window (more sensitive to changes)
    blink_rates_df = calculate_moving_window_blink_rate(df, window_size)
    
    print(f"Moving Window Analysis (Window Size: {window_size}s)")
    print("-" * 50)
    
    # Detect blink rate increases
    increases = detect_blink_rate_increases(blink_rates_df, threshold_increase=0.3)
    
    print("BLINK RATE INCREASE DETECTION")
    print("=" * 35)
    
    if increases:
        print(f"Detected {len(increases)} significant blink rate increases:")
        for i, increase in enumerate(increases):
            print(f"  Increase {i+1}:")
            print(f"    Time: {increase['timestamp']:.1f}s")
            print(f"    Window: {increase['window_start']:.1f}s - {increase['window_end']:.1f}s")
            print(f"    Previous rate: {increase['previous_rate']:.2f} blinks/second")
            print(f"    Current rate: {increase['current_rate']:.2f} blinks/second")
            print(f"    Increase: +{increase['increase']:.2f} blinks/second")
            print()
    else:
        print("No significant blink rate increases detected.")
        print()
    
    # Show sample of all blink rates
    print("SAMPLE BLINK RATES (every 3rd window):")
    print("-" * 40)
    for i in range(0, len(blink_rates_df), 3):
        row = blink_rates_df.iloc[i]
        print(f"Time {row['timestamp']:.1f}s: {row['blink_rate']:.2f} blinks/second")
    
    # Overall statistics
    print("\nOVERALL STATISTICS")
    print("=" * 20)
    print(f"Total analysis time: {df['timestamp'].max():.1f} seconds")
    print(f"Average blink rate: {blink_rates_df['blink_rate'].mean():.2f} blinks/second")
    print(f"Max blink rate: {blink_rates_df['blink_rate'].max():.2f} blinks/second")
    print(f"Min blink rate: {blink_rates_df['blink_rate'].min():.2f} blinks/second")
    
    return blink_rates_df, increases

def main():
    """Main function to demonstrate moving window blink detection"""
    
    print("MOVING WINDOW BLINK RATE DETECTION")
    print("=" * 40)
    print("This demo shows how to detect sudden changes")
    print("in blink rate using a sliding time window.")
    print()
    
    # Create sample CSV with varying blink patterns
    csv_filename = create_sample_csv()
    
    # Analyze moving window blink rates
    blink_rates_df, increases = analyze_moving_window_blinks(csv_filename)
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("The moving window approach detects blink rate increases:")
    print(f"• Detected {len(increases)} significant increases")
    print("• Each increase shows when blink rate jumped by +0.2 blinks/second")
    print("• This indicates potential stress or cognitive load changes")
    print()
    print("This method identifies when blink patterns change suddenly.")

if __name__ == "__main__":
    main()
