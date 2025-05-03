const express = require('express');
const router = express.Router();
const { SerialPort } = require('serialport');
const { ReadlineParser } = require('@serialport/parser-readline');

let port = null;
let parser = null;

async function connectToESP32() {
    try {
        // Check if port is already open
        if (port) {
            port.close();
        }

        port = new SerialPort({
            path: 'COM5',
            baudRate: 115200,
            dataBits: 8,
            parity: 'none',
            stopBits: 1,
            autoOpen: false  // Don't open automatically
        });

        // Open port with error handling
        await new Promise((resolve, reject) => {
            port.open((err) => {
                if (err) {
                    reject(err);
                } else {
                    resolve();
                }
            });
        });

        parser = port.pipe(new ReadlineParser({ delimiter: '\r\n' }));

        parser.on('data', (data) => {
            try {
                const readings = data.split(',');
                latestSensorData = {
                    temperature: parseFloat(readings[0]),
                    humidity: parseFloat(readings[1]),
                    ph: parseFloat(readings[2]),
                    tds: parseFloat(readings[3]),
                    timestamp: new Date().toISOString()
                };
                console.log('New readings:', latestSensorData);
            } catch (error) {
                console.error('Error parsing sensor data:', error);
            }
        });

        port.on('error', (err) => {
            console.error('Serial port error:', err);
            port = null;
        });

        console.log('Connected to ESP32 on COM5');  // Fixed log message
        return true;
    } catch (error) {
        console.error('Error connecting to ESP32:', error);
        return false;
    }
}

let latestSensorData = {
    temperature: 0,
    humidity: 0,
    ph: 0,
    tds: 0,
    timestamp: new Date().toISOString()
};

connectToESP32();

// Get latest sensor readings
router.get('/readings', (req, res) => {
    if (!port) {
        return res.status(503).json({ error: 'Device not connected' });
    }
    res.json(latestSensorData);
});

// Health check endpoint
router.get('/status', (req, res) => {
    res.json({ 
        connected: port !== null,
        lastReading: latestSensorData.timestamp
    });
});

// Manual reconnect endpoint
router.post('/reconnect', async (req, res) => {
    const connected = await connectToESP32();
    if (connected) {
        res.json({ message: 'Device reconnected successfully' });
    } else {
        res.status(503).json({ error: 'Failed to reconnect to device' });
    }
});

module.exports = router;