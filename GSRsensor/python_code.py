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
SERIAL_PORT = 'COM3'  # <<< IMPORTANT: CHANGE THIS TO YOUR ARDUINO'S PORT
BAUD_RATE = 9600
CSV_FILE = 'gsr_data.csv'

# --- Buffer Configuration ---
BUFFER_SIZE = 50 # NEW: Number of readings to collect before writing to file

# --- Filter Configuration ---
gsr_kalman_filter = SimpleKalmanFilter(mea_e=2, est_e=2, q=0.01)
baseline_kalman_filter = SimpleKalmanFilter(mea_e=10, est_e=10, q=0.001)

# --- Stress Detection Parameters ---
PHASIC_THRESHOLD = 13.0
REFRACTORY_PERIOD_MS = 2500

# --- Global Variables ---
last_event_time_ms = 0

def get_current_time_ms():
    """Returns the current time in milliseconds using a monotonic clock."""
    return int(time.monotonic() * 1000)

def main():
    global last_event_time_ms
    data_buffer = [] # NEW: Initialize an empty list to act as the buffer

    print("GSR Dynamic Baseline Stress Detection (with Buffered CSV Logging)")
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
                        
                        event_detected = ""
                        if current_time_ms - last_event_time_ms > REFRACTORY_PERIOD_MS:
                            if phasic_signal > PHASIC_THRESHOLD:
                                event_detected = "Stress Event"
                                print(f"*** {event_detected} detected at {timestamp} ***")
                                last_event_time_ms = current_time_ms
                        
                        # MODIFIED: Append the row to the buffer instead of writing directly
                        row_data = [timestamp, raw_value, f"{fast_signal:.2f}", f"{slow_baseline:.2f}", f"{phasic_signal:.2f}", event_detected]
                        data_buffer.append(row_data)

                        # NEW: Check if the buffer is full
                        if len(data_buffer) >= BUFFER_SIZE:
                            writer.writerows(data_buffer) # Use writerows to write all rows at once
                            print(f"--- Flushed {len(data_buffer)} rows to {CSV_FILE} ---")
                            data_buffer.clear() # Clear the buffer after writing

                    except (ValueError, TypeError):
                        print(f"Warning: Received non-numeric data: '{line}'")

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        # NEW: Ensure any remaining data in the buffer is saved on exit
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