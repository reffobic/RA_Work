/*
  GSR Sensor with DYNAMIC BASELINE and PHASIC THRESHOLD Detection

  This sketch uses a more robust method that directly measures the
  magnitude of the phasic signal (the difference between the fast signal
  and the slow baseline). This correctly identifies large spikes even
  if their initial rise is not extremely sharp.

  Required Library: "SimpleKalmanFilter" by Denys Sene.


  3.7.9

  
*/

#include <SimpleKalmanFilter.h>

// --- Pin Setup ---
const int GSR_PIN = A0;

// --- Filter Configuration ---
SimpleKalmanFilter gsrKalmanFilter(2, 2, 0.01);
SimpleKalmanFilter baselineKalmanFilter(10, 10, 0.001);

// --- Stress Detection Parameters (TUNE THESE) ---
// How high the fast signal needs to be above the slow baseline to trigger.
// This directly measures the "size" of the spike.
const float PHASIC_THRESHOLD = 13.0;
const int REFRACTORY_PERIOD = 2500; // Time (ms) to wait after an event.

// --- Global Variables ---
long lastEventTime = 0;

void setup() {
  Serial.begin(9600);
  Serial.println("GSR Dynamic Baseline Stress Detection (Phasic Threshold)");
  Serial.println("Monitoring will begin immediately...");
  Serial.println("---------------------------------");
}

void loop() {
  // --- Read and Filter the Sensor Data ---
  int rawValue = analogRead(GSR_PIN);
  double fastSignal = gsrKalmanFilter.updateEstimate(rawValue);
  double slowBaseline = baselineKalmanFilter.updateEstimate(rawValue);

  // --- Detection Logic ---
  detectStress(fastSignal, slowBaseline);
  
  delay(50);
}

void detectStress(double currentSignal, double currentBaseline) {
  // Optional: Uncomment the line below to see the signals for tuning.
  Serial.print("Baseline:"); Serial.print(currentBaseline); Serial.print(", Signal:"); Serial.println(currentSignal);

  // Calculate the current phasic signal (the "wave" on top of the "tide")
  double phasicSignal = currentSignal - currentBaseline;

  // Check if enough time has passed since the last event
  if (millis() - lastEventTime > REFRACTORY_PERIOD) {
    // Check if the phasic signal (the size of the spike) is large enough
    if (phasicSignal > PHASIC_THRESHOLD) {
      Serial.print("Stress event detected at timestamp (ms): ");
      Serial.println(millis());
      lastEventTime = millis(); // Reset the timer
    }
  }
}
