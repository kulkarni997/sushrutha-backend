#include <Wire.h>
#include "MAX30100_PulseOximeter.h"
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

const char* ssid = "Galaxy M55s 5G 8365";
const char* password = "sruuu0105";
const char* ws_host = "10.161.223.90";
const uint16_t ws_port = 8000;
const char* ws_path = "/ws/pulse";

#define REPORTING_PERIOD_MS 1000

PulseOximeter pox;
WebSocketsClient webSocket;
uint32_t tsLastReport = 0;

// Shared variables between cores
volatile float sharedBPM = 0;
volatile float sharedSpO2 = 0;

void onBeatDetected() {
  Serial.println("Beat!");
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] Disconnected");
      break;
    case WStype_CONNECTED:
      Serial.println("[WS] Connected to backend");
      break;
    case WStype_TEXT:
      Serial.printf("[WS] Got text: %s\n", payload);
      break;
    default:
      break;
  }
}

// Sensor task — runs on Core 0, never blocked
void sensorTask(void * parameter) {
  Serial.print("Initializing pulse oximeter on Core 0..");
  if (!pox.begin()) {
    Serial.println("FAILED");
    while(1) delay(1000);
  }
  Serial.println("SUCCESS");

  pox.setIRLedCurrent(MAX30100_LED_CURR_27_1MA);
  pox.setOnBeatDetectedCallback(onBeatDetected);

  for(;;) {
    pox.update();
    sharedBPM = pox.getHeartRate();
    sharedSpO2 = pox.getSpO2();
    delay(2);  // tiny delay to allow other tasks
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("=== ESP32 Booting ===");
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP: ");
  Serial.println(WiFi.localIP());

  webSocket.begin(ws_host, ws_port, ws_path);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);

  // Pin sensor to Core 0
  xTaskCreatePinnedToCore(
    sensorTask,    // function
    "sensorTask",  // name
    10000,         // stack size
    NULL,          // parameters
    1,             // priority
    NULL,          // task handle
    0              // core 0
  );

  Serial.println("Setup done. Sensor on Core 0, WS on Core 1.");
}

void loop() {
  webSocket.loop();

  if (millis() - tsLastReport > REPORTING_PERIOD_MS) {
    float bpm = sharedBPM;
    float spo2 = sharedSpO2;

    Serial.printf("BPM: %.2f | SpO2: %.0f%%\n", bpm, spo2);

    JsonDocument doc;
    doc["bpm"] = bpm;
    doc["spo2"] = spo2;
    doc["timestamp"] = millis();

    String payload;
    serializeJson(doc, payload);
    webSocket.sendTXT(payload);

    tsLastReport = millis();
  }
}