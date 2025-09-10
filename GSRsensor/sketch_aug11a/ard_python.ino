/*
  GSR Raw Data Sender for Python

  This sketch reads the raw value from the GSR sensor on pin A0
  and prints it to the serial port. All complex processing will be
  done in the Python script.
*/

const int GSR_PIN = A0;

void setup() {
  // Start serial communication at 9600 bits per second
  Serial.begin(9600);
}

void loop() {
  // Read the raw integer value from the sensor (0-1023)
  int rawValue = analogRead(GSR_PIN);

  // Print the value to the serial port, followed by a newline
  Serial.println(rawValue);

  // Delay to control the data rate, matching the original sketch
  delay(50);
}