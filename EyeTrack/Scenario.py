import sys
import time



import carla

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


# Start listening to the ego sensor
ego_sensor.listen(ego_sensor_callback)

# Keep the script running to allow listening
try:
    print("Listening to ego sensor... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping listener...")
    ego_sensor.stop()
