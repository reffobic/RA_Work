import serial
import time
import csv
from datetime import datetime

class SimpleKalmanFilter:
    """A simple Kalman filter implementation."""
    def __init__(self, mea_e, est_e, q):
        self._err_measure, self._err_estimate, self._q = mea_e, est_e, q
        self._current_estimate = 0
        self._last_estimate = 0
        self._kalman_gain = 0

    def update_estimate(self, mea):
        """Updates the filter estimate with a new measurement."""
        self._kalman_gain = self._err_estimate / (self._err_estimate + self._err_measure)
        self._current_estimate = self._last_estimate + self._kalman_gain * (mea - self._last_estimate)
        self._err_estimate = (1 - self._kalman_gain) * self._err_estimate + abs(self._last_estimate - self._current_estimate) * self._q
        self._last_estimate = self._current_estimate
        return self._current_estimate

# --- Configuration ---
SERIAL_PORT = '/dev/cu.usbmodem21301'  # <<< IMPORTANT: CHANGE THIS TO YOUR ARDUINO'S PORT
BAUD_RATE = 9600
CSV_FILE = 'gsr_data.csv'

# --- Buffer Configuration ---
BUFFER_SIZE = 50 #number of readings to collect before writing to file

# --- Filter Configuration ---
gsr_kalman_filter = SimpleKalmanFilter(mea_e=2, est_e=2, q=0.01)
baseline_kalman_filter = SimpleKalmanFilter(mea_e=10, est_e=10, q=0.001)

# --- Stress Detection Parameters ---
PHASIC_THRESHOLD = 13.0  # Same as Arduino sketch
REFRACTORY_PERIOD_MS = 2500

# --- Global Variables ---
last_event_time_ms = 0

def get_current_time_ms():
    """Returns the current time in milliseconds using a monotonic clock."""
    return int(time.monotonic() * 1000)

def detect_stress(current_signal, current_baseline):
    """Detect stress events using the same logic as the Arduino sketch."""
    global last_event_time_ms
    
    # Calculate the current phasic signal (the "wave" on top of the "tide")
    phasic_signal = current_signal - current_baseline
    
    # Check if enough time has passed since the last event
    current_time_ms = get_current_time_ms()
    if current_time_ms - last_event_time_ms > REFRACTORY_PERIOD_MS:
        # Check if the phasic signal (the size of the spike) is large enough
        if phasic_signal > PHASIC_THRESHOLD:
            print(f"*** Stress event detected at timestamp (ms): {current_time_ms} ***")
            print(f"    Phasic signal: {phasic_signal:.2f} (threshold: {PHASIC_THRESHOLD})")
            last_event_time_ms = current_time_ms
            return "Stress Event"
    
    return ""

def main():
    global last_event_time_ms
    data_buffer = [] # NEW: Initialize an empty list to act as the buffer

    print("GSR Dynamic Baseline Stress Detection (Phasic Threshold)")
    print("Monitoring will begin immediately...")
    print("-----------------------------------------------------------------")

    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Connected to Arduino on {SERIAL_PORT}")
        print(f"Logging data to {CSV_FILE} in chunks of {BUFFER_SIZE}. Press Ctrl+C to stop.")

    except serial.SerialException as e:
        print(f"FATAL ERROR: Could not open port '{SERIAL_PORT}'. {e}")
        return

    try:
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                writer.writerow(['Timestamp', 'RawValue', 'FastSignal', 'SlowBaseline', 'PhasicSignal', 'Event'])

            while True:
                line = arduino.readline().decode('utf-8').strip()

                if line:
                    try:
                        raw_value = int(line)
                        fast_signal = gsr_kalman_filter.update_estimate(raw_value)
                        slow_baseline = baseline_kalman_filter.update_estimate(raw_value)
                        phasic_signal = fast_signal - slow_baseline
                        current_time_ms = get_current_time_ms()
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        
                        # Use the same stress detection logic as Arduino sketch
                        event_detected = detect_stress(fast_signal, slow_baseline)
                        
                        # Optional: Uncomment the line below to see the signals for tuning
                        print(f"Baseline:{slow_baseline:.2f}, Signal:{fast_signal:.2f}")
                        
                        # Append the row to the buffer instead of writing directly
                        row_data = [timestamp, raw_value, f"{fast_signal:.2f}", f"{slow_baseline:.2f}", f"{phasic_signal:.2f}", event_detected]
                        data_buffer.append(row_data)

                        # Check if the buffer is full
                        if len(data_buffer) >= BUFFER_SIZE:
                            writer.writerows(data_buffer) # Use writerows to write all rows at once
                            print(f"--- Flushed {len(data_buffer)} rows to {CSV_FILE} ---")
                            data_buffer.clear() # Clear the buffer after writing

                    except (ValueError, TypeError):
                        print(f"Warning: Received non-numeric data: '{line}'")

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        # Ensure any remaining data in the buffer is saved on exit
        if data_buffer:
            print(f"--- Saving remaining {len(data_buffer)} readings... ---")
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data_buffer)
        
        if 'arduino' in locals() and arduino.is_open:
            arduino.close()
            print("Serial port closed.")

if __name__ == '__main__':
    main()