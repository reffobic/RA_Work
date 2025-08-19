import sys
import time
import csv
import threading
from collections import deque
from datetime import datetime
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import carla

# Configuration for buffering
BUFFER_SIZE = 100  # Number of data points to collect before writing
WRITE_INTERVAL = 5.0  # Seconds between writes 
CSV_FILENAME = f"eye_tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Thread-safe buffer for storing eye tracking data
data_buffer = deque(maxlen=BUFFER_SIZE)
buffer_lock = threading.Lock()
last_write_time = time.time()

# Stress detection variables
stress_features = {
    'blink_count': 0,
    'scanpath_length': 0.0,
    'gaze_points': [],
    'session_start_time': time.time(),
    'last_blink_time': 0,
    'total_fixations': 0,
    'saccade_count': 0,
    'avg_fixation_duration': 0.0,
    'pupil_diameter_variance': 0.0,
    'gaze_velocity': 0.0
}

# Stress detection parameters (based on research findings)
STRESS_THRESHOLDS = {
    'blink_rate_threshold': 0.3,  # blinks per second (high stress = more blinks)
    'scanpath_length_threshold': 1000.0,  # pixels (high stress = longer scanpath)
    'fixation_duration_threshold': 200.0,  # ms (high stress = shorter fixations)
    'pupil_variance_threshold': 0.5,  # mm² (high stress = more variance)
    'gaze_velocity_threshold': 50.0  # degrees/second (high stress = faster gaze)
}

# SVM model for stress classification (simplified version)
stress_model = None
scaler = StandardScaler()

# CSV headers
CSV_HEADERS = [
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
    'gaze_vergence', 'focus_actor_dist', 'focus_actor_name',
    'location_x', 'location_y', 'location_z',
    'rotation_pitch', 'rotation_yaw', 'rotation_roll',
    # Stress detection features
    'blink_count', 'scanpath_length', 'gaze_velocity', 'pupil_variance'
]

def write_buffer_to_csv():
    """Write buffered data to CSV file"""
    global data_buffer, last_write_time
    
    with buffer_lock:
        if not data_buffer:
            return
        
        # Get all data from buffer
        data_to_write = list(data_buffer)
        data_buffer.clear()
    
    # Write to CSV
    try:
        with open(CSV_FILENAME, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
            
            # Write header if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()
            
            # Write all buffered data
            for data_row in data_to_write:
                writer.writerow(data_row)
        
        print(f"Wrote {len(data_to_write)} data points to {CSV_FILENAME}")
        last_write_time = time.time()
        
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def detect_blink(left_openness, right_openness, timestamp):
    """Detect blink based on eye openness"""
    global stress_features
    
    # Blink detection: both eyes close significantly
    blink_threshold = 0.3
    if left_openness < blink_threshold and right_openness < blink_threshold:
        # Check if enough time has passed since last blink (avoid double counting)
        if timestamp - stress_features['last_blink_time'] > 0.1:  # 100ms minimum between blinks
            stress_features['blink_count'] += 1
            stress_features['last_blink_time'] = timestamp
            return True
    return False

def calculate_scanpath_length(gaze_x, gaze_y):
    """Calculate scanpath length from gaze coordinates"""
    global stress_features
    
    current_point = (gaze_x, gaze_y)
    stress_features['gaze_points'].append(current_point)
    
    # Keep only last 100 points to avoid memory issues
    if len(stress_features['gaze_points']) > 100:
        stress_features['gaze_points'] = stress_features['gaze_points'][-100:]
    
    # Calculate total scanpath length
    if len(stress_features['gaze_points']) > 1:
        total_length = 0
        for i in range(1, len(stress_features['gaze_points'])):
            prev_point = stress_features['gaze_points'][i-1]
            curr_point = stress_features['gaze_points'][i]
            distance = np.sqrt((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2)
            total_length += distance
        stress_features['scanpath_length'] = total_length
    
    return stress_features['scanpath_length']

def calculate_gaze_velocity(gaze_x, gaze_y, timestamp):
    """Calculate gaze velocity"""
    global stress_features
    
    if len(stress_features['gaze_points']) >= 2:
        # Calculate velocity from last two points
        prev_point = stress_features['gaze_points'][-2]
        curr_point = (gaze_x, gaze_y)
        distance = np.sqrt((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2)
        time_diff = 0.016  # Assuming 60Hz sampling rate
        stress_features['gaze_velocity'] = distance / time_diff
    
    return stress_features['gaze_velocity']

def analyze_stress_levels():
    """Analyze current stress levels based on eye tracking features"""
    global stress_features, STRESS_THRESHOLDS
    
    current_time = time.time()
    session_duration = current_time - stress_features['session_start_time']
    
    # Calculate stress indicators
    blink_rate = stress_features['blink_count'] / session_duration if session_duration > 0 else 0
    scanpath_length = stress_features['scanpath_length']
    gaze_velocity = stress_features['gaze_velocity']
    
    # Stress scoring (0-100 scale)
    stress_score = 0
    
    # Blink rate analysis (high stress = more blinks)
    if blink_rate > STRESS_THRESHOLDS['blink_rate_threshold']:
        stress_score += 25
        print(f"High blink rate detected: {blink_rate:.2f} blinks/sec")
    
    # Scanpath length analysis (high stress = longer scanpath)
    if scanpath_length > STRESS_THRESHOLDS['scanpath_length_threshold']:
        stress_score += 25
        print(f"Long scanpath detected: {scanpath_length:.1f} pixels")
    
    # Gaze velocity analysis (high stress = faster gaze)
    if gaze_velocity > STRESS_THRESHOLDS['gaze_velocity_threshold']:
        stress_score += 25
        print(f"High gaze velocity detected: {gaze_velocity:.1f} deg/sec")
    
    # Pupil diameter variance (high stress = more variance)
    if stress_features['pupil_diameter_variance'] > STRESS_THRESHOLDS['pupil_variance_threshold']:
        stress_score += 25
        print(f"High pupil variance detected: {stress_features['pupil_diameter_variance']:.2f}")
    
    # Determine stress level
    if stress_score >= 75:
        stress_level = "HIGH STRESS"
        print("HIGH STRESS DETECTED")
    elif stress_score >= 50:
        stress_level = "MODERATE STRESS"
        print("MODERATE STRESS DETECTED")
    elif stress_score >= 25:
        stress_level = "LOW STRESS"
        print("LOW STRESS DETECTED")
    else:
        stress_level = "NO STRESS"
        print("NO STRESS DETECTED")
    
    return stress_score, stress_level

def add_to_buffer(data):
    """Add eye tracking data to buffer with stress detection"""
    global data_buffer, last_write_time, stress_features
    
    # Detect blink
    detect_blink(data.left_eye_openness, data.right_eye_openness, data.timestamp)
    
    # Calculate scanpath length
    scanpath_length = calculate_scanpath_length(data.left_gaze_dir.x, data.left_gaze_dir.y)
    
    # Calculate gaze velocity
    gaze_velocity = calculate_gaze_velocity(data.left_gaze_dir.x, data.left_gaze_dir.y, data.timestamp)
    
    # Update pupil diameter variance
    avg_pupil_diam = (data.left_pupil_diam + data.right_pupil_diam) / 2.0
    if len(data_buffer) > 0:
        # Calculate variance from recent samples
        recent_pupils = [row['avg_pupil_diam'] for row in list(data_buffer)[-10:]]
        recent_pupils.append(avg_pupil_diam)
        stress_features['pupil_diameter_variance'] = np.var(recent_pupils)
    
    # Create data row with stress features
    data_row = {
        'timestamp': data.timestamp,
        'timestamp_carla': data.timestamp_carla,
        'timestamp_device': data.timestamp_device,
        'timestamp_stream': data.timestamp_stream,
        'left_pupil_diam': data.left_pupil_diam,
        'right_pupil_diam': data.right_pupil_diam,
        'avg_pupil_diam': avg_pupil_diam,
        'left_gaze_dir_x': data.left_gaze_dir.x,
        'left_gaze_dir_y': data.left_gaze_dir.y,
        'left_gaze_dir_z': data.left_gaze_dir.z,
        'right_gaze_dir_x': data.right_gaze_dir.x,
        'right_gaze_dir_y': data.right_gaze_dir.y,
        'right_gaze_dir_z': data.right_gaze_dir.z,
        'left_gaze_origin_x': data.left_gaze_origin.x,
        'left_gaze_origin_y': data.left_gaze_origin.y,
        'left_gaze_origin_z': data.left_gaze_origin.z,
        'right_gaze_origin_x': data.right_gaze_origin.x,
        'right_gaze_origin_y': data.right_gaze_origin.y,
        'right_gaze_origin_z': data.right_gaze_origin.z,
        'left_eye_openness': data.left_eye_openness,
        'right_eye_openness': data.right_eye_openness,
        'gaze_valid': data.gaze_valid,
        'left_gaze_valid': data.left_gaze_valid,
        'right_gaze_valid': data.right_gaze_valid,
        'left_pupil_posn_x': data.left_pupil_posn.x,
        'left_pupil_posn_y': data.left_pupil_posn.y,
        'left_pupil_posn_valid': data.left_pupil_posn_valid,
        'right_pupil_posn_x': data.right_pupil_posn.x,
        'right_pupil_posn_y': data.right_pupil_posn.y,
        'right_pupil_posn_valid': data.right_pupil_posn_valid,
        'gaze_vergence': data.gaze_vergence,
        'focus_actor_dist': data.focus_actor_dist,
        'focus_actor_name': data.focus_actor_name,
        'location_x': data.transform.location.x,
        'location_y': data.transform.location.y,
        'location_z': data.transform.location.z,
        'rotation_pitch': data.transform.rotation.pitch,
        'rotation_yaw': data.transform.rotation.yaw,
        'rotation_roll': data.transform.rotation.roll,
        # Stress detection features
        'blink_count': stress_features['blink_count'],
        'scanpath_length': scanpath_length,
        'gaze_velocity': gaze_velocity,
        'pupil_variance': stress_features['pupil_diameter_variance']
    }
    
    with buffer_lock:
        data_buffer.append(data_row)
    
    # Check if we should write to CSV
    current_time = time.time()
    if (len(data_buffer) >= BUFFER_SIZE or 
        (current_time - last_write_time) >= WRITE_INTERVAL):
        # Use threading to avoid blocking the main callback
        threading.Thread(target=write_buffer_to_csv, daemon=True).start()

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)

# Get the world
world = client.get_world()

actors = world.get_actors()

for actor in actors:
    print(actor)
# Get the ego sensor actor
ego_sensor = world.get_actor(87)

# Confirm it's the correct sensor
print(f"Found ego sensor: {ego_sensor.id}, type: {ego_sensor.type_id}")

# Define the callback function to receive sensor data
def ego_sensor_callback(data):
    print("----- Ego Sensor Data Received -----")
    print(f"Type: {type(data)}")
    print(data)

    print(f"Location: {data.transform.location}")
    print(f"Rotation: {data.transform.rotation}")

    # Access pupil diameters
    print(f"Left Pupil Diameter: {data.left_pupil_diam}")
    print(f"Right Pupil Diameter: {data.right_pupil_diam}")

    # Optional: average if needed
    avg_diameter = (data.left_pupil_diam + data.right_pupil_diam) / 2.0
    print(f"Average (Gaze) Diameter: {avg_diameter}")
    
    # Add data to buffer for CSV recording with stress detection
    add_to_buffer(data)
    
    # Analyze stress levels every 5 seconds (every 300 data points at 60Hz)
    if len(data_buffer) % 300 == 0:
        stress_score, stress_level = analyze_stress_levels()
        print(f"Current Stress Score: {stress_score}/100 - {stress_level}")
        print(f"Blink Count: {stress_features['blink_count']}")
        print(f"Scanpath Length: {stress_features['scanpath_length']:.1f}")
        print(f"Gaze Velocity: {stress_features['gaze_velocity']:.1f}")
        print(f"Pupil Variance: {stress_features['pupil_diameter_variance']:.3f}")
        print("-" * 50)

# Start listening to the ego sensor
ego_sensor.listen(ego_sensor_callback)

print(f"Eye tracking data will be saved to: {CSV_FILENAME}")
print(f"Buffer size: {BUFFER_SIZE} data points")
print(f"Write interval: {WRITE_INTERVAL} seconds")

# Keep the script running to allow listening
try:
    print("Listening to ego sensor... Press Ctrl+C to stop.")
    print("Stress detection is active - monitoring stress indicators...")
    print("Stress analysis will be displayed every 5 seconds")
    print("-" * 50)
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping listener...")
    # Write any remaining data in buffer
    write_buffer_to_csv()
    ego_sensor.stop()
    
    # Final stress analysis
    print("\n" + "="*60)
    print("FINAL STRESS ANALYSIS REPORT")
    print("="*60)
    stress_score, stress_level = analyze_stress_levels()
    print(f"Final Stress Score: {stress_score}/100")
    print(f"Stress Level: {stress_level}")
    print(f"Total Blinks: {stress_features['blink_count']}")
    print(f"Final Scanpath Length: {stress_features['scanpath_length']:.1f}")
    print(f"Max Gaze Velocity: {stress_features['gaze_velocity']:.1f}")
    print(f"Average Pupil Variance: {stress_features['pupil_diameter_variance']:.3f}")
    print("="*60)
    print("Data collection stopped.")
