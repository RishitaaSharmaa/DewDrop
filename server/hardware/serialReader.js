import { SerialPort } from "serialport";
import { ReadlineParser } from "@serialport/parser-readline";

class SerialReader {
  constructor(port = null, baudRate = 115200) {
    this.port = port;
    this.baudRate = baudRate;
    this.serialPort = null;
    this.parser = null;
    this.isConnected = false;
    this.onDataCallback = null;
    this.onConnectionChange = null;
    this.retryCount = 0;
    this.maxRetries = 5;
    this.latestReadings = {
      temperature: 0,
      humidity: 0,
      ph: 0,
      tds: 0,
    };
  }

  async listPorts() {
    try {
      const ports = await SerialPort.list();
      console.log("Available serial ports:");
      ports.forEach((port) => {
        console.log(
          `- ${port.path} (${port.manufacturer || "Unknown manufacturer"})`
        );
      });
      return ports;
    } catch (error) {
      console.error("Error listing ports:", error);
      return [];
    }
  }

  async connect() {
    try {
      // List available ports first
      const ports = await this.listPorts();

      // If no port specified, try to find ESP32
      if (!this.port) {
        const esp32Port = ports.find(
          (port) =>
            port.manufacturer?.toLowerCase().includes("silicon labs") ||
            port.manufacturer?.toLowerCase().includes("ftdi") ||
            port.manufacturer?.toLowerCase().includes("wch.cn") ||
            port.path.toLowerCase().includes("com5") // Also check for COM5 specifically
        );

        if (esp32Port) {
          this.port = esp32Port.path;
          console.log(`Found ESP32 on port: ${this.port}`);
        } else {
          // If no port found but COM5 exists, use it
          const com5Port = ports.find(
            (port) => port.path.toLowerCase() === "com5"
          );
          if (com5Port) {
            this.port = "COM5";
            console.log("Using COM5 as default port");
          } else {
            throw new Error(
              "No ESP32 found. Please specify the correct COM port."
            );
          }
        }
      }

      // Close existing connection if any
      if (this.serialPort && this.serialPort.isOpen) {
        await this.disconnect();
      }

      this.serialPort = new SerialPort({
        path: this.port,
        baudRate: this.baudRate,
        autoOpen: false,
      });

      this.parser = this.serialPort.pipe(
        new ReadlineParser({ delimiter: "\r\n" })
      );

      this.serialPort.on("open", () => {
        console.log(`Serial port ${this.port} opened successfully`);
        this.isConnected = true;
        this.retryCount = 0;
        if (this.onConnectionChange) {
          this.onConnectionChange(true);
        }
      });

      this.serialPort.on("close", () => {
        console.log(`Serial port ${this.port} closed`);
        this.isConnected = false;
        if (this.onConnectionChange) {
          this.onConnectionChange(false);
        }
      });

      this.serialPort.on("error", async (err) => {
        console.error(`Serial port error on ${this.port}:`, err.message);
        this.isConnected = false;
        if (this.onConnectionChange) {
          this.onConnectionChange(false);
        }

        // Retry connection if we haven't exceeded max retries
        if (this.retryCount < this.maxRetries) {
          this.retryCount++;
          console.log(
            `Retrying connection (attempt ${this.retryCount}/${this.maxRetries})...`
          );
          await new Promise((resolve) => setTimeout(resolve, 2000)); // Wait 2 seconds
          await this.connect();
        } else {
          console.error(
            "Max retry attempts reached. Please check your connection."
          );
        }
      });

      this.parser.on("data", (data) => {
        try {
          console.log("Raw data received from ESP32:", data);
          // Parse the incoming data string
          // Expected format: "T:25.5,H:60.2,pH:7.2,TDS:120"
          const values = {};
          data.split(",").forEach((pair) => {
            const [key, value] = pair.split(":");
            switch (key.trim()) {
              case "T":
                values.temperature = parseFloat(value) || 0;
                break;
              case "H":
                values.humidity = parseFloat(value) || 0;
                break;
              case "pH":
                values.ph = parseFloat(value) || 0;
                break;
              case "TDS":
                values.tds = parseFloat(value) || 0;
                break;
            }
          });

          // Check if all required fields are present, regardless of their values
          const requiredFields = ["temperature", "humidity", "ph", "tds"];
          const missingFields = requiredFields.filter(
            (field) => !(field in values)
          );

          if (missingFields.length > 0) {
            console.warn("Missing required fields:", missingFields);
            return;
          }

          console.log("Parsed values:", values);

          // Update latest readings
          this.latestReadings = values;

          if (this.onDataCallback) {
            this.onDataCallback(JSON.stringify(values));
          }
        } catch (error) {
          console.error("Error parsing sensor data:", error);
          console.error("Raw data that caused error:", data);
        }
      });

      await this.serialPort.open();
    } catch (error) {
      console.error("Failed to connect to serial port:", error);
      throw error;
    }
  }

  async disconnect() {
    if (this.serialPort) {
      return new Promise((resolve) => {
        if (this.serialPort.isOpen) {
          this.serialPort.close(() => {
            console.log("Serial port closed");
            resolve();
          });
        } else {
          resolve();
        }
      });
    }
  }

  onData(callback) {
    this.onDataCallback = callback;
  }

  onConnectionStatusChange(callback) {
    this.onConnectionChange = callback;
  }

  getConnectionStatus() {
    return this.isConnected;
  }

  getLatestReadings() {
    return this.latestReadings;
  }
}

export default SerialReader;
