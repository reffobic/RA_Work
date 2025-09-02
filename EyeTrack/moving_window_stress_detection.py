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
    # Gaze coordinates are realistic for 1920x1080 screen
    sample_data = [
        # timestamp, left_eye_openness, right_eye_openness, left_pupil_diam, right_pupil_diam, gaze_x, gaze_y
        # Period 1: Normal blinking (0-3 seconds)
        [0.0, 0.8, 0.9, 3.2, 3.1, 960, 540],  # Normal, small movements
        [0.3, 0.1, 0.1, 3.1, 3.0, 965, 542],  # BLINK, slight move
        [0.6, 0.8, 0.9, 3.3, 3.2, 968, 544],  # Normal
        [0.9, 0.9, 0.8, 3.4, 3.3, 970, 545],  # Normal
        [1.2, 0.1, 0.1, 3.2, 3.1, 972, 546],  # BLINK
        [1.5, 0.8, 0.9, 3.5, 3.4, 975, 548],  # Normal
        [1.8, 0.9, 0.8, 3.6, 3.5, 977, 550],  # Normal
        [2.1, 0.1, 0.1, 3.3, 3.2, 980, 552],  # BLINK
        [2.4, 0.8, 0.9, 3.7, 3.6, 982, 553],  # Normal
        [2.7, 0.9, 0.8, 3.8, 3.7, 984, 554],  # Normal
        
        # Period 2: High stress - rapid blinking (3-6 seconds)
        [3.0, 0.1, 0.1, 3.9, 3.8, 1200, 400],  # BLINK, large move starts
        [3.2, 0.8, 0.9, 4.0, 3.9, 1400, 300],  # Normal, big jump
        [3.4, 0.1, 0.1, 4.1, 4.0, 1600, 600],  # BLINK
        [3.6, 0.8, 0.9, 4.2, 4.1, 1800, 200],  # Normal
        [3.8, 0.1, 0.1, 4.3, 4.2, 200, 800],  # BLINK
        [4.0, 0.8, 0.9, 4.4, 4.3, 400, 100],  # Normal
        [4.2, 0.1, 0.1, 4.5, 4.4, 600, 900],  # BLINK
        [4.4, 0.8, 0.9, 4.6, 4.5, 800, 50],  # Normal
        [4.6, 0.1, 0.1, 4.7, 4.6, 1000, 950],  # BLINK
        [4.8, 0.8, 0.9, 4.8, 4.7, 1200, 150],  # Normal
        [5.0, 0.1, 0.1, 4.9, 4.8, 1400, 1000],  # BLINK
        [5.2, 0.8, 0.9, 5.0, 4.9, 1600, 200],  # Normal, still large moves
        [5.4, 0.1, 0.1, 5.1, 5.0, 1800, 800],  # BLINK
        [5.6, 0.8, 0.9, 5.2, 5.1, 200, 400],  # Normal
        [5.8, 0.1, 0.1, 5.3, 5.2, 400, 600],  # BLINK
        [6.0, 0.8, 0.9, 5.4, 5.3, 600, 800],  # Normal
        
        # Period 3: Return to normal (6-9 seconds)
        [6.3, 0.1, 0.1, 5.5, 5.4, 800, 500],  # BLINK, movements calm down
        [6.6, 0.8, 0.9, 5.6, 5.5, 820, 510],  # Normal
        [6.9, 0.9, 0.8, 5.7, 5.6, 840, 520],  # Normal
        [7.2, 0.1, 0.1, 5.8, 5.7, 860, 530],  # BLINK
        [7.5, 0.8, 0.9, 5.9, 5.8, 880, 540],  # Normal
        [7.8, 0.9, 0.8, 6.0, 5.9, 900, 550],  # Normal
        [8.1, 0.1, 0.1, 6.1, 6.0, 920, 560],  # BLINK
        [8.4, 0.8, 0.9, 6.2, 6.1, 940, 570],  # Normal
        [8.7, 0.9, 0.8, 6.3, 6.2, 960, 580],  # Normal
        [9.0, 0.1, 0.1, 6.4, 6.3, 980, 590],  # BLINK
        [9.3, 0.8, 0.9, 6.5, 6.4, 1000, 600],  # Normal
        
        # Period 4: Moderate stress (9-12 seconds)
        [9.6, 0.1, 0.1, 6.6, 6.5, 1200, 300],  # BLINK, moderate stress with medium jumps
        [9.8, 0.8, 0.9, 6.7, 6.6, 1180, 320],  # Normal
        [10.0, 0.1, 0.1, 6.8, 6.7, 1220, 280],  # BLINK
        [10.2, 0.8, 0.9, 6.9, 6.8, 1190, 310],  # Normal
        [10.4, 0.1, 0.1, 7.0, 6.9, 1230, 270],  # BLINK
        [10.6, 0.8, 0.9, 7.1, 7.0, 1200, 300],  # Normal
        [10.8, 0.1, 0.1, 7.2, 7.1, 1240, 260],  # BLINK
        [11.0, 0.8, 0.9, 7.3, 7.2, 1210, 290],  # Normal
        [11.2, 0.1, 0.1, 7.4, 7.3, 1250, 250],  # BLINK
        [11.4, 0.8, 0.9, 7.5, 7.4, 1220, 280],  # Normal
        [11.6, 0.1, 0.1, 7.6, 7.5, 1260, 240],  # BLINK
        [11.8, 0.8, 0.9, 7.7, 7.6, 1230, 270],  # Normal
        [12.0, 0.1, 0.1, 7.8, 7.7, 1260, 240],  # BLINK
        [12.2, 0.8, 0.9, 7.9, 7.8, 1230, 270],  # Normal
        
        # Period 5: Final normal period (12-15 seconds)
        [12.5, 0.1, 0.1, 8.0, 7.9, 1000, 420],  # BLINK, return to small moves
        [12.8, 0.8, 0.9, 8.1, 8.0, 1002, 422],  # Normal
        [13.1, 0.9, 0.8, 8.2, 8.1, 1004, 424],  # Normal
        [13.4, 0.1, 0.1, 8.3, 8.2, 1006, 426],  # BLINK
        [13.7, 0.8, 0.9, 8.4, 8.3, 1008, 428],  # Normal
        [14.0, 0.9, 0.8, 8.5, 8.4, 1010, 430],  # Normal
        [14.3, 0.1, 0.1, 8.6, 8.5, 1012, 432],  # BLINK
        [14.6, 0.8, 0.9, 8.7, 8.6, 1014, 433],  # Normal
        [14.9, 0.9, 0.8, 8.8, 8.7, 1016, 434],  # Normal
        [15.2, 0.1, 0.1, 8.9, 8.8, 1018, 435],  # BLINK
        [15.5, 0.8, 0.9, 9.0, 8.9, 1020, 436],  # Normal
    ]
    
    csv_filename = "moving_window_demo.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'left_eye_openness', 'right_eye_openness', 
                        'left_pupil_diam', 'right_pupil_diam', 'gaze_x', 'gaze_y'])
        for row in sample_data:
            writer.writerow(row)
    
    print(f"Created sample CSV: {csv_filename}")
    print("Data pattern:")
    print("  0-3s: Normal blinking (3 blinks)")
    print("  3-6s: High stress - rapid blinking (8 blinks)")
    print("  6-9s: Return to normal (4 blinks)")
    print("  9-12s: Moderate stress (6 blinks)")
    print("  12-15s: Final normal period (4 blinks)")
    print("  Gaze coordinates: 1920×1080 screen, 32 px/degree")
    print()
    
    return csv_filename

def detect_blink(timestamp, left_openness, right_openness):
    blink_threshold = 0.3
    return left_openness < blink_threshold and right_openness < blink_threshold

def calculate_moving_window_metrics(df, window_size=2.0):
    
    results = []
    blink_threshold = 0.3
    refractory_period = 0.1  # Minimum time between blinks
    # Hive Pro specifications
    PIXELS_PER_DEGREE = 32.0  # 1920px ÷ 60° = 32 px/degree
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
    SAMPLING_RATE = 120  # Hz
    
    for i, row in df.iterrows():
        current_time = row['timestamp']
        window_start = current_time - window_size
        
        # Only analyze windows that start at or after the first data point
        if window_start < 0:
            results.append({
                'timestamp': current_time,
                'blink_rate': 0.0,
                'pupil_variance': 0.0,
                'window_start': window_start,
                'window_end': current_time,
                'blinks_in_window': 0,
                'window_duration': 0.0,
                'scanpath_length': 0.0,
                'gaze_velocity_deg_s': 0.0
            })
            continue
        
        # Get all data points within the current window
        window_data = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= current_time)]
        
        if len(window_data) < 2:  # Need at least 2 points for meaningful calculation
            results.append({
                'timestamp': current_time,
                'blink_rate': 0.0,
                'pupil_variance': 0.0,
                'window_start': window_start,
                'window_end': current_time,
                'blinks_in_window': 0,
                'window_duration': 0.0,
                'scanpath_length': 0.0,
                'gaze_velocity_deg_s': 0.0
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
        
        # Calculate pupil diameter variance for this window
        left_pupils = window_data['left_pupil_diam'].values
        right_pupils = window_data['right_pupil_diam'].values
        avg_pupils = (left_pupils + right_pupils) / 2
        pupil_variance = np.var(avg_pupils) if len(avg_pupils) > 1 else 0.0
        
        # Calculate scanpath length (sum of distances between consecutive gaze points) in pixels
        gaze_points = window_data[['gaze_x', 'gaze_y']].values
        if len(gaze_points) >= 2:
            diffs = np.diff(gaze_points, axis=0)
            step_distances = np.sqrt(np.sum(diffs**2, axis=1))
            scanpath_length = float(np.sum(step_distances))
        else:
            scanpath_length = 0.0
        
        # Calculate gaze velocity (degrees/second), using Hive Pro pixels-per-degree
        if window_duration > 0 and PIXELS_PER_DEGREE > 0:
            angular_path_deg = scanpath_length / PIXELS_PER_DEGREE
            gaze_velocity = angular_path_deg / window_duration
        else:
            gaze_velocity = 0.0
        
        results.append({
            'timestamp': current_time,
            'blink_rate': blink_rate,
            'pupil_variance': pupil_variance,
            'window_start': window_start,
            'window_end': current_time,
            'blinks_in_window': len(blinks_in_window),
            'window_duration': window_duration,
            'scanpath_length': scanpath_length,
            'gaze_velocity_deg_s': gaze_velocity
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


def detect_metric_increases(metrics_df, column_name, threshold_increase):
    """Generic increase detector for a numeric metric over windows."""
    events = []
    previous_value = None
    for i, row in metrics_df.iterrows():
        current_value = row[column_name]
        timestamp = row['timestamp']
        if i == 0:
            previous_value = current_value
            continue
        if previous_value is not None and current_value > previous_value + threshold_increase:
            events.append({
                'timestamp': timestamp,
                'current_value': current_value,
                'previous_value': previous_value,
                'increase': current_value - previous_value,
                'window_start': row['window_start'],
                'window_end': row['window_end']
            })
        previous_value = current_value
    return events

def analyze_moving_window_blinks(csv_filename):
    
    print(f"Analyzing moving window blink rates from: {csv_filename}")
    print("=" * 60)
    
    # Read CSV
    df = pd.read_csv(csv_filename)
    
    print("Sample data (first 10 rows):")
    print(df.head(10).to_string(index=False))
    print()
    
    # Calculate moving window metrics
    window_size = 2.0  # 2-second window (more sensitive to changes)
    metrics_df = calculate_moving_window_metrics(df, window_size)
    
    print(f"Moving Window Analysis (Window Size: {window_size}s)")
    print("-" * 50)
    
    # Detect blink rate increases
    increases = detect_blink_rate_increases(metrics_df, threshold_increase=0.3)
    
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
    
    # Detect scanpath length increases (threshold: 200 pixels = ~6.25 degrees)
    scanpath_increases = detect_metric_increases(metrics_df, 'scanpath_length', threshold_increase=200.0)
    print()
    print("SCANPATH LENGTH INCREASE DETECTION")
    print("=" * 40)
    if scanpath_increases:
        print(f"Detected {len(scanpath_increases)} significant scanpath length increases:")
        for i, ev in enumerate(scanpath_increases):
            print(f"  Increase {i+1}:")
            print(f"    Time: {ev['timestamp']:.1f}s")
            print(f"    Window: {ev['window_start']:.1f}s - {ev['window_end']:.1f}s")
            print(f"    Previous length: {ev['previous_value']:.1f} px")
            print(f"    Current length: {ev['current_value']:.1f} px")
            print(f"    Increase: +{ev['increase']:.1f} px")
            print()
    else:
        print("No significant scanpath length increases detected.")

    # Detect gaze velocity increases (threshold: 1.0 deg/s = typical saccade velocity)
    gaze_vel_increases = detect_metric_increases(metrics_df, 'gaze_velocity_deg_s', threshold_increase=1.0)
    print()
    print("GAZE VELOCITY INCREASE DETECTION")
    print("=" * 38)
    if gaze_vel_increases:
        print(f"Detected {len(gaze_vel_increases)} significant gaze velocity increases:")
        for i, ev in enumerate(gaze_vel_increases):
            print(f"  Increase {i+1}:")
            print(f"    Time: {ev['timestamp']:.1f}s")
            print(f"    Window: {ev['window_start']:.1f}s - {ev['window_end']:.1f}s")
            print(f"    Previous velocity: {ev['previous_value']:.2f} deg/s")
            print(f"    Current velocity: {ev['current_value']:.2f} deg/s")
            print(f"    Increase: +{ev['increase']:.2f} deg/s")
            print()
    else:
        print("No significant gaze velocity increases detected.")

    # Show all metrics by window
    print()
    print("METRICS BY WINDOW:")
    print("-" * 60)
    print("Time(s) | Blink Rate | Pupil Variance | Scanpath(px) | Gaze Vel(deg/s)")
    print("-" * 60)
    for i, row in metrics_df.iterrows():
        print(f"{row['timestamp']:6.1f}s | {row['blink_rate']:9.2f} | {row['pupil_variance']:13.4f} | {row['scanpath_length']:11.1f} | {row['gaze_velocity_deg_s']:15.2f}")
    
    # Overall statistics
    print("\nOVERALL STATISTICS")
    print("=" * 20)
    print(f"Total analysis time: {df['timestamp'].max():.1f} seconds")
    print(f"Average blink rate: {metrics_df['blink_rate'].mean():.2f} blinks/second")
    print(f"Max blink rate: {metrics_df['blink_rate'].max():.2f} blinks/second")
    print(f"Min blink rate: {metrics_df['blink_rate'].min():.2f} blinks/second")
    print(f"Average pupil variance: {metrics_df['pupil_variance'].mean():.4f} mm²")
    print(f"Max pupil variance: {metrics_df['pupil_variance'].max():.4f} mm²")
    print(f"Min pupil variance: {metrics_df['pupil_variance'].min():.4f} mm²")
    print(f"Average scanpath length: {metrics_df['scanpath_length'].mean():.1f} px")
    print(f"Max scanpath length: {metrics_df['scanpath_length'].max():.1f} px")
    print(f"Min scanpath length: {metrics_df['scanpath_length'].min():.1f} px")
    print(f"Average gaze velocity: {metrics_df['gaze_velocity_deg_s'].mean():.2f} deg/s")
    print(f"Max gaze velocity: {metrics_df['gaze_velocity_deg_s'].max():.2f} deg/s")
    print(f"Min gaze velocity: {metrics_df['gaze_velocity_deg_s'].min():.2f} deg/s")
    
    return metrics_df, increases, scanpath_increases, gaze_vel_increases

def main():
    """Main function to demonstrate moving window blink detection"""
    
    print("MOVING WINDOW BLINK RATE DETECTION")
    print("=" * 40)
    print("This demo shows how to detect sudden changes")
    print("in blink rate using a sliding time window.")
    print()
    
    # Create sample CSV with varying blink patterns
    csv_filename = create_sample_csv()
    
    # Analyze moving window metrics
    metrics_df, increases, scanpath_increases, gaze_vel_increases = analyze_moving_window_blinks(csv_filename)
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("The moving window approach detects sudden increases in metrics:")
    print(f"• Blink rate increases: {len(increases)}")
    print(f"• Scanpath length increases: {len(scanpath_increases)}")
    print(f"• Gaze velocity increases: {len(gaze_vel_increases)}")
    print()
    print("This method identifies when blink patterns change suddenly.")

if __name__ == "__main__":
    main()
