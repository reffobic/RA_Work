#!/usr/bin/env python3
import csv
import numpy as np
import pandas as pd
from datetime import datetime

def create_sample_csv():
    
    sample_data = [
        # timestamp, left_eye_openness, right_eye_openness, left_pupil_diam, right_pupil_diam
        [0.0, 0.8, 0.9, 3.2, 3.1],  # Normal 
        [0.1, 0.7, 0.8, 3.3, 3.2],  # Normal 
        [0.2, 0.1, 0.1, 3.1, 3.0],  # BLINK 
        [0.3, 0.8, 0.9, 3.4, 3.3],  # Normal 
        [0.4, 0.9, 0.8, 3.5, 3.4],  # Normal 
        [0.5, 0.2, 0.1, 3.2, 3.1],  # BLINK 
        [0.6, 0.8, 0.9, 3.6, 3.5],  # Normal 
        [0.7, 0.1, 0.1, 3.7, 3.6],  # BLINK  
        [0.8, 0.1, 0.2, 3.3, 3.2],  # BLINK 
        [0.9, 0.8, 0.9, 3.8, 3.7],  # Normal 
    ]
    
    csv_filename = "simple_stress_demo.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['timestamp', 'left_eye_openness', 'right_eye_openness', 
                        'left_pupil_diam', 'right_pupil_diam'])
        
        for row in sample_data:
            writer.writerow(row)
    
    print(f"Created sample CSV: {csv_filename}")
    print("Sample data:")
    for i, row in enumerate(sample_data):
        print(f"  Entry {i+1}: Time={row[0]}s, Left Eye={row[1]}, Right Eye={row[2]}, "
              f"Left Pupil={row[3]}mm, Right Pupil={row[4]}mm")
    print()
    
    return csv_filename

def detect_blinks(left_openness, right_openness, timestamps):
    
    blinks = []
    blink_threshold = 0.3  
    refractory_period = 0.2  # Minimum 200ms between blinks
    
    last_blink_time = -1
    
    for i, (timestamp, left, right) in enumerate(zip(timestamps, left_openness, right_openness)):
        # Check if both eyes are closed (blink)
        if left < blink_threshold and right < blink_threshold:
            # Check if enough time has passed since last blink
            if timestamp - last_blink_time > refractory_period:
                blinks.append({
                    'timestamp': timestamp,
                    'index': i,
                    'left_openness': left,
                    'right_openness': right
                })
                last_blink_time = timestamp
    
    return blinks

def calculate_blink_rate(blinks, total_duration):
    if total_duration > 0:
        return len(blinks) / total_duration
    return 0.0

def calculate_pupil_variance(left_pupil, right_pupil):
    avg_pupil = (left_pupil + right_pupil) / 2
    return np.var(avg_pupil)

def analyze_stress_from_csv(csv_filename):
    
    print(f"Analyzing stress indicators from: {csv_filename}")
    print("=" * 50)
    
    # Read CSV
    df = pd.read_csv(csv_filename)
    
    print("CSV Data:")
    print(df.to_string(index=False))
    print()
    
    # Extract data
    timestamps = df['timestamp'].values
    left_openness = df['left_eye_openness'].values
    right_openness = df['right_eye_openness'].values
    left_pupil = df['left_pupil_diam'].values
    right_pupil = df['right_pupil_diam'].values
    
    # Calculate total duration
    total_duration = timestamps[-1] - timestamps[0]
    
    print(f"Analysis Period: {total_duration:.1f} seconds")
    print()
    
    # 1. BLINK DETECTION
    print("1. BLINK DETECTION")
    print("-" * 20)
    
    blinks = detect_blinks(left_openness, right_openness, timestamps)
    
    print(f"Detected {len(blinks)} blinks:")
    for i, blink in enumerate(blinks):
        print(f"  Blink {i+1}: Time={blink['timestamp']:.1f}s, "
              f"Left Eye={blink['left_openness']:.1f}, Right Eye={blink['right_openness']:.1f}")
    
    blink_rate = calculate_blink_rate(blinks, total_duration)
    print(f"Blink Rate: {blink_rate:.2f} blinks/second")
    print()
    
    # 2. PUPIL DIAMETER VARIANCE
    print("2. PUPIL DIAMETER VARIANCE")
    print("-" * 30)
    
    # Calculate average pupil diameter for each sample
    avg_pupil_diameters = (left_pupil + right_pupil) / 2
    
    print("Average Pupil Diameters:")
    for i, (timestamp, avg_pupil) in enumerate(zip(timestamps, avg_pupil_diameters)):
        print(f"  Sample {i+1}: Time={timestamp:.1f}s, Avg Pupil={avg_pupil:.1f}mm")
    
    pupil_variance = calculate_pupil_variance(left_pupil, right_pupil)
    print(f"Pupil Diameter Variance: {pupil_variance:.3f} mm²")
    print()
    
    return {
        'blink_rate': blink_rate,
        'pupil_variance': pupil_variance,
        'blinks': blinks
    }

def main():
    
    print("SIMPLE STRESS DETECTION DEMO")
    print("=" * 40)
    print("This demo shows how to calculate stress indicators")
    print("using only essential eye tracking data.")
    print()
    
    # Create sample CSV
    csv_filename = create_sample_csv()
    
    # Analyze stress
    results = analyze_stress_from_csv(csv_filename)
    
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"• Total blinks detected: {len(results['blinks'])}")
    print(f"• Blink rate: {results['blink_rate']:.2f} blinks/second")
    print(f"• Pupil diameter variance: {results['pupil_variance']:.3f} mm²")
    print()
    print("The sample data shows 3 blinks in 0.9 seconds.")
    print("These calculations can be used for stress detection analysis.")

if __name__ == "__main__":
    main()
