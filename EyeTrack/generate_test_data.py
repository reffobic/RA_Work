#!/usr/bin/env python3
"""
Test Data Generator for Eye Tracking
Generates realistic eye tracking data to test HDF5 storage and analysis
"""

import numpy as np
import pandas as pd
import h5py
from datetime import datetime
import time
import threading
from collections import deque
import random

# Configuration
BUFFER_SIZE = 50
WRITE_INTERVAL = 2.0
HDF5_FILENAME = f"test_eye_tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
DURATION_SECONDS = 30  # Generate 30 seconds of data
SAMPLE_RATE = 60  # 60 Hz eye tracking data

# Thread-safe buffer
data_buffer = deque(maxlen=BUFFER_SIZE)
buffer_lock = threading.Lock()
last_write_time = time.time()

# Data columns (same as real eye tracking data)
data_columns = [
    'timestamp', 'timestamp_carla', 'timestamp_device', 'timestamp_stream',
    'left_pupil_diam', 'right_pupil_diam', 'avg_pupil_diam',
    'left_gaze_dir_x', 'left_gaze_dir_y', 'left_gaze_dir_z',
    'right_gaze_dir_x', 'right_gaze_dir_y', 'right_gaze_dir_z',
    'left_gaze_origin_x', 'left_gaze_origin_y', 'left_gaze_origin_z',
    'right_gaze_origin_x', 'right_gaze_origin_y', 'right_gaze_origin_z',
    'left_eye_openness', 'right_eye_openness',
    'gaze_valid', 'left_gaze_valid', 'right_gaze_valid',
    'left_pupil_posn_x', 'left_pupil_posn_y', 'left_pupil_posn_valid',
    'right_pupil_posn_x', 'right_pupil_posn_y', 'right_pupil_posn_valid',
    'gaze_vergence', 'focus_actor_dist',
    'location_x', 'location_y', 'location_z',
    'rotation_pitch', 'rotation_yaw', 'rotation_roll'
]

def generate_realistic_eye_data(timestamp):
    """Generate realistic eye tracking data"""
    
    # Base pupil diameter with natural variation
    base_pupil = 4.5 + np.sin(timestamp * 0.1) * 0.5 + np.random.normal(0, 0.1)
    left_pupil = base_pupil + np.random.normal(0, 0.05)
    right_pupil = base_pupil + np.random.normal(0, 0.05)
    
    # Gaze direction (simulating looking around)
    gaze_x = np.sin(timestamp * 0.3) * 0.3 + np.random.normal(0, 0.05)
    gaze_y = np.cos(timestamp * 0.2) * 0.2 + np.random.normal(0, 0.05)
    gaze_z = 1.0 + np.random.normal(0, 0.02)
    
    # Eye openness (blinking simulation)
    blink_prob = 0.02  # 2% chance of blink per sample
    if random.random() < blink_prob:
        left_openness = right_openness = 0.1 + np.random.normal(0, 0.05)
    else:
        left_openness = 0.9 + np.random.normal(0, 0.05)
        right_openness = 0.9 + np.random.normal(0, 0.05)
    
    # Gaze validity (occasional tracking loss)
    gaze_valid = random.random() > 0.05  # 95% valid
    left_gaze_valid = gaze_valid and random.random() > 0.02
    right_gaze_valid = gaze_valid and random.random() > 0.02
    
    # Focus distance (simulating depth changes)
    focus_dist = 5.0 + np.sin(timestamp * 0.15) * 2.0 + np.random.normal(0, 0.3)
    
    # Position and rotation (simulating head movement)
    location_x = np.sin(timestamp * 0.1) * 0.5
    location_y = np.cos(timestamp * 0.1) * 0.5
    location_z = 1.7 + np.random.normal(0, 0.02)
    
    rotation_yaw = np.sin(timestamp * 0.05) * 10.0
    rotation_pitch = np.cos(timestamp * 0.03) * 5.0
    rotation_roll = np.random.normal(0, 1.0)
    
    return {
        'timestamp': timestamp,
        'timestamp_carla': timestamp * 1000,  # Convert to milliseconds
        'timestamp_device': timestamp * 1000 + np.random.normal(0, 1),
        'timestamp_stream': timestamp * 1000 + np.random.normal(0, 1),
        'left_pupil_diam': max(2.0, left_pupil),
        'right_pupil_diam': max(2.0, right_pupil),
        'avg_pupil_diam': (left_pupil + right_pupil) / 2.0,
        'left_gaze_dir_x': gaze_x + np.random.normal(0, 0.02),
        'left_gaze_dir_y': gaze_y + np.random.normal(0, 0.02),
        'left_gaze_dir_z': gaze_z + np.random.normal(0, 0.01),
        'right_gaze_dir_x': gaze_x + np.random.normal(0, 0.02),
        'right_gaze_dir_y': gaze_y + np.random.normal(0, 0.02),
        'right_gaze_dir_z': gaze_z + np.random.normal(0, 0.01),
        'left_gaze_origin_x': 0.03 + np.random.normal(0, 0.001),
        'left_gaze_origin_y': 0.0 + np.random.normal(0, 0.001),
        'left_gaze_origin_z': 0.0 + np.random.normal(0, 0.001),
        'right_gaze_origin_x': -0.03 + np.random.normal(0, 0.001),
        'right_gaze_origin_y': 0.0 + np.random.normal(0, 0.001),
        'right_gaze_origin_z': 0.0 + np.random.normal(0, 0.001),
        'left_eye_openness': max(0.0, min(1.0, left_openness)),
        'right_eye_openness': max(0.0, min(1.0, right_openness)),
        'gaze_valid': gaze_valid,
        'left_gaze_valid': left_gaze_valid,
        'right_gaze_valid': right_gaze_valid,
        'left_pupil_posn_x': np.random.normal(0, 0.1),
        'left_pupil_posn_y': np.random.normal(0, 0.1),
        'left_pupil_posn_valid': left_gaze_valid,
        'right_pupil_posn_x': np.random.normal(0, 0.1),
        'right_pupil_posn_y': np.random.normal(0, 0.1),
        'right_pupil_posn_valid': right_gaze_valid,
        'gaze_vergence': abs(left_pupil - right_pupil) * 10,
        'focus_actor_dist': max(0.1, focus_dist),

        'location_x': location_x,
        'location_y': location_y,
        'location_z': location_z,
        'rotation_pitch': rotation_pitch,
        'rotation_yaw': rotation_yaw,
        'rotation_roll': rotation_roll
    }

def write_buffer_to_hdf5():
    """Write buffered data to HDF5 file"""
    global data_buffer, last_write_time
    
    with buffer_lock:
        if not data_buffer:
            return
        
        data_to_write = list(data_buffer)
        data_buffer.clear()
    
    # Convert to numpy arrays
    data_dict = {col: [] for col in data_columns}
    
    for data_row in data_to_write:
        for col in data_columns:
            data_dict[col].append(data_row[col])
    
    arrays = {col: np.array(data_dict[col]) for col in data_columns}
    
    # Write to HDF5
    try:
        with h5py.File(HDF5_FILENAME, 'a') as hf:
            for col in data_columns:
                if col in hf:
                    current_size = hf[col].shape[0]
                    new_size = current_size + len(arrays[col])
                    hf[col].resize((new_size,))
                    hf[col][current_size:] = arrays[col]
                else:

                        hf.create_dataset(col, data=arrays[col], 
                                        maxshape=(None,), 
                                        compression='gzip', 
                                        compression_opts=9,
                                        chunks=True)
            
            # Store metadata
            if 'metadata' not in hf:
                metadata_group = hf.create_group('metadata')
                metadata_group.attrs['created'] = datetime.now().isoformat()
                metadata_group.attrs['description'] = 'Simulated eye tracking data for testing'
                metadata_group.attrs['data_columns'] = data_columns
                metadata_group.attrs['buffer_size'] = BUFFER_SIZE
                metadata_group.attrs['write_interval'] = WRITE_INTERVAL
                metadata_group.attrs['sample_rate'] = SAMPLE_RATE
                metadata_group.attrs['duration'] = DURATION_SECONDS
        
        print(f"Wrote {len(data_to_write)} data points to {HDF5_FILENAME}")
        last_write_time = time.time()
        
    except Exception as e:
        print(f"Error writing to HDF5: {e}")

def add_to_buffer(data):
    """Add data to buffer"""
    global data_buffer, last_write_time
    
    with buffer_lock:
        data_buffer.append(data)
    
    # Check if we should write
    current_time = time.time()
    if (len(data_buffer) >= BUFFER_SIZE or 
        (current_time - last_write_time) >= WRITE_INTERVAL):
        threading.Thread(target=write_buffer_to_hdf5, daemon=True).start()

def generate_test_data():
    """Generate test eye tracking data"""
    print(f"🎯 Generating {DURATION_SECONDS} seconds of test eye tracking data...")
    print(f"📊 Sample rate: {SAMPLE_RATE} Hz")
    print(f"📁 Output file: {HDF5_FILENAME}")
    print(f"⏱️  Buffer size: {BUFFER_SIZE} data points")
    print(f"🔄 Write interval: {WRITE_INTERVAL} seconds")
    print("-" * 60)
    
    start_time = time.time()
    sample_interval = 1.0 / SAMPLE_RATE
    total_samples = int(DURATION_SECONDS * SAMPLE_RATE)
    
    for i in range(total_samples):
        timestamp = i * sample_interval
        
        # Generate realistic eye tracking data
        data = generate_realistic_eye_data(timestamp)
        
        # Add to buffer
        add_to_buffer(data)
        
        # Progress indicator
        if i % (SAMPLE_RATE * 5) == 0:  # Every 5 seconds
            progress = (i / total_samples) * 100
            elapsed = time.time() - start_time
            print(f"⏳ Progress: {progress:.1f}% ({i}/{total_samples} samples, {elapsed:.1f}s elapsed)")
        
        # Small delay to simulate real-time data collection
        time.sleep(sample_interval)
    
    # Write any remaining data
    write_buffer_to_hdf5()
    
    print("-" * 60)
    print(f"✅ Test data generation complete!")
    print(f"📁 File created: {HDF5_FILENAME}")
    print(f"📊 Total samples: {total_samples}")
    print(f"⏱️  Total time: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    try:
        generate_test_data()
        
        print("\n🎉 Now you can test the analysis tools:")
        print(f"python quick_view.py {HDF5_FILENAME}")
        print(f"python analyze_data.py {HDF5_FILENAME}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Data generation interrupted by user")
        write_buffer_to_hdf5()
    except Exception as e:
        print(f"❌ Error: {e}")
