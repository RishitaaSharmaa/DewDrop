import { SerialPort } from "serialport";
import { ReadlineParser } from "@serialport/parser-readline";

class SerialReader {
  constructor(port = "COM5", baudRate = 9600) {
    this.port = port;
    this.baudRate = baudRate;
    this.serialPort = null;
    this.parser = null;
    this.isConnected = false;
    this.onDataCallback = null;
    this.onConnectionChange = null;
    this.latestReadings = {
      temperature: 0,
      humidity: 0,
      ph: 0,
      tds: 0,
    };
  }

  connect() {
    this.serialPort = new SerialPort({
      path: this.port,
      baudRate: this.baudRate,
      autoOpen: false,
    });

    this.parser = this.serialPort.pipe(
      new ReadlineParser({ delimiter: "\r\n" })
    );

    this.serialPort.on("open", () => {
      console.log(`Serial port ${this.port} opened`);
      this.isConnected = true;
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

    this.serialPort.on("error", (err) => {
      console.error("Error:", err.message);
      this.isConnected = false;
      if (this.onConnectionChange) {
        this.onConnectionChange(false);
      }
    });

    this.parser.on("data", (data) => {
      try {
        // Parse the incoming data string
        // Expected format: "T:25.5,H:60.2,pH:7.2,TDS:120"
        const values = {};
        data.split(",").forEach((pair) => {
          const [key, value] = pair.split(":");
          switch (key) {
            case "T":
              values.temperature = parseFloat(value);
              break;
            case "H":
              values.humidity = parseFloat(value);
              break;
            case "pH":
              values.ph = parseFloat(value);
              break;
            case "TDS":
              values.tds = parseFloat(value);
              break;
          }
        });

        // Update latest readings
        this.latestReadings = values;

        if (this.onDataCallback) {
          this.onDataCallback(JSON.stringify(values));
        }
      } catch (error) {
        console.error("Error parsing sensor data:", error);
      }
    });

    this.serialPort.open();
  }

  disconnect() {
    if (this.serialPort && this.serialPort.isOpen) {
      this.serialPort.close();
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
