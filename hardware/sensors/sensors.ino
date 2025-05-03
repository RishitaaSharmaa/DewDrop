#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>

// OLED display settings
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Pin definitions
#define PH_SENSOR_PIN 34
#define TDS_SENSOR_PIN 35
#define DHT_PIN 4
#define DHT_TYPE DHT22

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();

  // Initialize OLED display
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 OLED initialization failed.");
    while (true); // Stop here if OLED not found
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println("Starting sensors...");
  display.display();
  delay(2000);
}

void loop() {
  // 🌡 DHT22 readings
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  if (isnan(temperature) || isnan(humidity)) {
    temperature = 0.0;
    humidity = 0.0;
  }

  // ⚗ pH reading (still hardcoded)
  int phRaw = analogRead(PH_SENSOR_PIN);
  float pH = 7.0;  // Set constant for now due to 4095 raw value issue

  // 💧 TDS reading
  int tdsRaw = analogRead(TDS_SENSOR_PIN);
  float tdsVoltage = (tdsRaw / 4095.0) * 3.3;
  float tds = (tdsVoltage * 133.42);  // Simplified conversion factor
  if (tds < 10) tds = 0;              // Ignore very low readings as noise

  // Send data in the format expected by the server
  Serial.print("T:");
  Serial.print(temperature, 2);
  Serial.print(",H:");
  Serial.print(humidity, 2);
  Serial.print(",pH:");
  Serial.print(pH, 2);
  Serial.print(",TDS:");
  Serial.println(tds, 2);

  // 🖥 OLED Display
  display.clearDisplay();
  display.setCursor(0, 0);
  display.print("pH: "); display.println(pH, 2);
  display.print("TDS: "); display.print(tds, 2); display.println(" ppm");
  display.print("Temp: "); display.print(temperature, 2); display.println(" C");
  display.print("Humidity: "); display.print(humidity, 2); display.println(" %");
  display.display();

  delay(2000);
}