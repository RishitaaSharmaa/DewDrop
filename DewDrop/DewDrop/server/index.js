import express from "express";
import cors from "cors";
import { WebSocketServer } from "ws";
import SerialReader from "./hardware/serialReader.js";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import http from "http";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = 3000;

// Enable CORS
app.use(cors());

// Create HTTP server
const server = http.createServer(app);

// Create WebSocket server
const wss = new WebSocketServer({
  server,
  path: "/ws",
  clientTracking: true,
});

const serialReader = new SerialReader("COM5", 115200);

// Store all connected clients
const clients = new Set();

// Handle WebSocket connections
wss.on("connection", (ws, req) => {
  console.log("New client connected from:", req.socket.remoteAddress);
  clients.add(ws);

  // Send initial connection status
  ws.send(
    JSON.stringify({
      type: "connection",
      status: serialReader.getConnectionStatus(),
    })
  );

  // Send latest readings if available
  const latestReadings = serialReader.getLatestReadings();
  if (latestReadings) {
    ws.send(
      JSON.stringify({
        type: "data",
        data: JSON.stringify(latestReadings),
      })
    );
  }

  ws.on("close", () => {
    console.log("Client disconnected");
    clients.delete(ws);
  });

  ws.on("error", (error) => {
    console.error("WebSocket error:", error);
    clients.delete(ws);
  });
});

// Add a test endpoint
app.get("/", (req, res) => {
  res.send("Server is running");
});

// Add an endpoint to get latest readings
app.get("/api/readings", (req, res) => {
  const readings = serialReader.getLatestReadings();
  res.json(readings);
});

// Start HTTP server
server.listen(port, "0.0.0.0", () => {
  console.log(`Server is running on http://localhost:${port}`);
  console.log(`WebSocket server is running on ws://localhost:${port}/ws`);
});

// Start serial reader with retry logic
async function startSerialReader() {
  try {
    await serialReader.connect();
    console.log("Serial connection established successfully");
  } catch (error) {
    console.error("Failed to initialize serial connection:", error);
    setTimeout(startSerialReader, 5000);
  }
}

startSerialReader();

// Handle serial data
serialReader.onData((data) => {
  try {
    console.log("Raw serial data received:", data);
    const readings = JSON.parse(data);
    console.log("Parsed readings:", readings);

    // Format the readings
    const formattedReadings = {
      tds: parseFloat(readings.tds) || 0,
      ph: parseFloat(readings.ph) || 0,
      temperature: parseFloat(readings.temperature) || 0,
      humidity: parseFloat(readings.humidity) || 0,
    };

    console.log("Formatted readings:", formattedReadings);

    // Create the message to send to clients
    const message = JSON.stringify({
      type: "data",
      data: JSON.stringify(formattedReadings),
    });

    // Broadcast to all connected clients
    let sentCount = 0;
    const clientsToRemove = new Set();

    clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        try {
          client.send(message);
          sentCount++;
          console.log("Message sent to client:", message);
        } catch (error) {
          console.error("Error sending message to client:", error);
          clientsToRemove.add(client);
        }
      } else {
        clientsToRemove.add(client);
      }
    });

    // Remove disconnected clients
    clientsToRemove.forEach((client) => {
      clients.delete(client);
    });

    console.log(`Sent to ${sentCount} clients`);
  } catch (error) {
    console.error("Error processing sensor data:", error);
    console.error("Raw data that caused error:", data);
  }
});

// Handle connection status changes
serialReader.onConnectionStatusChange((isConnected) => {
  console.log("Serial connection status changed:", isConnected);
  const message = JSON.stringify({
    type: "connection",
    status: isConnected,
  });

  const clientsToRemove = new Set();

  clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      try {
        client.send(message);
      } catch (error) {
        console.error("Error sending connection status:", error);
        clientsToRemove.add(client);
      }
    } else {
      clientsToRemove.add(client);
    }
  });

  // Remove disconnected clients
  clientsToRemove.forEach((client) => {
    clients.delete(client);
  });
});
