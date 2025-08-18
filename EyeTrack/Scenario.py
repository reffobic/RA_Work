import sys
import time
import threading
from collections import deque
from datetime import datetime
import numpy as np
import pandas as pd
import h5py

import carla

# Configuration for buffering
BUFFER_SIZE = 100  # Number of data points to collect before writing
WRITE_INTERVAL = 5.0  # Seconds between writes (even if buffer isn't full)
HDF5_FILENAME = f"eye_tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

# Thread-safe buffer for storing eye tracking data
data_buffer = deque(maxlen=BUFFER_SIZE)
buffer_lock = threading.Lock()
last_write_time = time.time()

# Data structure for efficient storage
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
    'gaze_vergence', 'focus_actor_dist', 'focus_actor_name',
    'location_x', 'location_y', 'location_z',
    'rotation_pitch', 'rotation_yaw', 'rotation_roll'
]

def write_buffer_to_hdf5():
    """Write buffered data to HDF5 file"""
    global data_buffer, last_write_time
    
    with buffer_lock:
        if not data_buffer:
            return
        
        # Get all data from buffer
        data_to_write = list(data_buffer)
        data_buffer.clear()
    
    # Convert to numpy arrays for efficient storage
    data_dict = {col: [] for col in data_columns}
    
    for data_row in data_to_write:
        for col in data_columns:
            data_dict[col].append(data_row[col])
    
    # Convert to numpy arrays
    arrays = {col: np.array(data_dict[col]) for col in data_columns}
    
    # Write to HDF5
    try:
        with h5py.File(HDF5_FILENAME, 'a') as hf:
            # Create or extend datasets
            for col in data_columns:
                if col in hf:
                    # Extend existing dataset
                    current_size = hf[col].shape[0]
                    new_size = current_size + len(arrays[col])
                    hf[col].resize((new_size,))
                    hf[col][current_size:] = arrays[col]
                else:
                    # Create new dataset with compression
                    hf.create_dataset(col, data=arrays[col], 
                                    maxshape=(None,), 
                                    compression='gzip', 
                                    compression_opts=9,
                                    chunks=True)
            
            # Store metadata
            if 'metadata' not in hf:
                metadata_group = hf.create_group('metadata')
                metadata_group.attrs['created'] = datetime.now().isoformat()
                metadata_group.attrs['description'] = 'Eye tracking data from CARLA simulation'
                metadata_group.attrs['data_columns'] = data_columns
                metadata_group.attrs['buffer_size'] = BUFFER_SIZE
                metadata_group.attrs['write_interval'] = WRITE_INTERVAL
        
        print(f"Wrote {len(data_to_write)} data points to {HDF5_FILENAME}")
        last_write_time = time.time()
        
    except Exception as e:
        print(f"Error writing to HDF5: {e}")

def add_to_buffer(data):
    """Add eye tracking data to buffer"""
    global data_buffer, last_write_time
    
    # Create data row
    data_row = {
        'timestamp': data.timestamp,
        'timestamp_carla': data.timestamp_carla,
        'timestamp_device': data.timestamp_device,
        'timestamp_stream': data.timestamp_stream,
        'left_pupil_diam': data.left_pupil_diam,
        'right_pupil_diam': data.right_pupil_diam,
        'avg_pupil_diam': (data.left_pupil_diam + data.right_pupil_diam) / 2.0,
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
        'rotation_roll': data.transform.rotation.roll
    }
    
    with buffer_lock:
        data_buffer.append(data_row)
    
    # Check if we should write to HDF5
    current_time = time.time()
    if (len(data_buffer) >= BUFFER_SIZE or 
        (current_time - last_write_time) >= WRITE_INTERVAL):
        # Use threading to avoid blocking the main callback
        threading.Thread(target=write_buffer_to_hdf5, daemon=True).start()

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


# Available attributes: ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'brake_input', 'camera_location', 'camera_rotation', 'current_gear_input', 'focus_actor_dist', 'focus_actor_name', 'focus_actor_pt', 'frame', 'frame_number', 'framesequence', 'gaze_dir', 'gaze_origin', 'gaze_valid', 'gaze_vergence', 'handbrake_input', 'left_eye_openness', 'left_eye_openness_valid', 'left_gaze_dir', 'left_gaze_origin', 'left_gaze_valid', 'left_pupil_diam', 'left_pupil_posn', 'left_pupil_posn_valid', 'right_eye_openness', 'right_eye_openness_valid', 'right_gaze_dir', 'right_gaze_origin', 'right_gaze_valid', 'right_pupil_diam', 'right_pupil_posn', 'right_pupil_posn_valid', 'steering_input', 'throttle_input', 'timestamp', 'timestamp_carla', 'timestamp_device', 'timestamp_stream', 'transform']

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
    
    # Add data to buffer for HDF5 recording
    add_to_buffer(data)

# Start listening to the ego sensor
ego_sensor.listen(ego_sensor_callback)

print(f"Eye tracking data will be saved to: {HDF5_FILENAME}")
print(f"Buffer size: {BUFFER_SIZE} data points")
print(f"Write interval: {WRITE_INTERVAL} seconds")

# Keep the script running to allow listening
try:
    print("Listening to ego sensor... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping listener...")
    # Write any remaining data in buffer
    write_buffer_to_hdf5()
    ego_sensor.stop()
    print("Data collection stopped.")
