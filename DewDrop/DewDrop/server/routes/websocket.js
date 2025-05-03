const WebSocket = require("ws");
const SerialReader = require("../../client/hardware/sensors/serialReader");

const wss = new WebSocket.Server({ port: 8080 });
const serialReader = new SerialReader();

// Store all connected clients
const clients = new Set();

wss.on("connection", (ws) => {
  console.log("New client connected");
  clients.add(ws);

  // Send initial connection status
  ws.send(
    JSON.stringify({
      type: "connection",
      status: serialReader.getConnectionStatus(),
    })
  );

  ws.on("close", () => {
    console.log("Client disconnected");
    clients.delete(ws);
  });
});

// Start serial reader
serialReader.connect();

// Handle serial data
serialReader.onData((data) => {
  const message = JSON.stringify({
    type: "data",
    data: data, // Data is already JSON stringified in SerialReader
  });

  // Broadcast to all connected clients
  clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
});

// Handle connection status changes
serialReader.onConnectionStatusChange((isConnected) => {
  const message = JSON.stringify({
    type: "connection",
    status: isConnected,
  });

  clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
});

console.log("WebSocket server is running on port 8080");
