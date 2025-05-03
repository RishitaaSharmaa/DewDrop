import React, { useState, useEffect } from 'react';
import { Droplet, Thermometer, Activity, Droplets } from 'lucide-react';

const WaterQualityMetrics = () => {
    const [sensorData, setSensorData] = useState({
        tds: 0,
        ph: 0,
        temperature: 0,
        humidity: 0
    });
    const [isConnected, setIsConnected] = useState(false);
    const [ws, setWs] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        const connectWebSocket = () => {
            const websocket = new WebSocket('ws://localhost:3000/ws');

            websocket.onopen = () => {
                console.log('WebSocket connection established');
                setIsConnected(true);
                setError(null);
            };

            websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    console.log('Received WebSocket message:', message);

                    if (message.type === 'data') {
                        const sensorReadings = JSON.parse(message.data);
                        console.log('Updating sensor data:', sensorReadings);

                        setSensorData(prevData => ({
                            ...prevData,
                            tds: parseFloat(sensorReadings.tds) || 0,
                            ph: parseFloat(sensorReadings.ph) || 0,
                            temperature: parseFloat(sensorReadings.temperature) || 0,
                            humidity: parseFloat(sensorReadings.humidity) || 0
                        }));
                    } else if (message.type === 'connection') {
                        console.log('Connection status update:', message.status);
                        setIsConnected(message.status);
                    }
                } catch (error) {
                    console.error('Error processing WebSocket message:', error);
                    setError('Error processing data from server');
                }
            };

            websocket.onclose = () => {
                console.log('WebSocket connection closed');
                setIsConnected(false);
                setError('Connection lost. Attempting to reconnect...');
                // Attempt to reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };

            websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                setIsConnected(false);
                setError('Connection error. Please check if the server is running.');
            };

            setWs(websocket);
        };

        connectWebSocket();

        return () => {
            if (ws) {
                ws.close();
            }
        };
    }, []);

    // Add the missing helper functions
    const getQualityLevel = (value, type) => {
        switch (type) {
            case 'tds':
                return value < 50 ? 'Excellent' : value < 100 ? 'Good' : value < 300 ? 'Fair' : 'Poor';
            case 'ph':
                return (value > 6.5 && value < 8.5) ? 'Excellent' : (value > 6 && value < 9) ? 'Good' : 'Poor';
            default:
                return 'Unknown';
        }
    };

    const getColorClass = (quality) => {
        switch (quality) {
            case 'Excellent':
                return 'text-green-600';
            case 'Good':
                return 'text-blue-600';
            case 'Fair':
                return 'text-yellow-600';
            case 'Poor':
                return 'text-red-600';
            default:
                return 'text-gray-600';
        }
    };

    return (
        <div className="bg-white rounded-lg shadow-md p-2 col-span-1 gap-1">
            <h3 className="text-lg font-semibold mb-2">Water Quality Metrics</h3>
            <div className="bg-white p-6 rounded-lg shadow-md border-2">
                <div className="flex items-center justify-between">
                    <div>
                        <p className="text-gray-500">TDS</p>
                        <p className="text-2xl font-bold">{sensorData.tds.toFixed(2)} ppm</p>
                    </div>
                    <Droplet className="text-blue-500" size={24} />
                </div>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md border-2">
                <div className="flex items-center justify-between">
                    <div>
                        <p className="text-gray-500">pH</p>
                        <p className="text-2xl font-bold">{sensorData.ph.toFixed(2)}</p>
                    </div>
                    <Activity className="text-green-500" size={24} />
                </div>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md border-2">
                <div className="flex items-center justify-between">
                    <div>
                        <p className="text-gray-500">Temperature</p>
                        <p className="text-2xl font-bold">{sensorData.temperature.toFixed(2)}°C</p>
                    </div>
                    <Thermometer className="text-red-500" size={24} />
                </div>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md border-2">
                <div className="flex items-center justify-between">
                    <div>
                        <p className="text-gray-500">Humidity</p>
                        <p className="text-2xl font-bold">{sensorData.humidity.toFixed(2)}%</p>
                    </div>
                    <Droplets className="text-purple-500" size={24} />
                </div>
            </div>
            <div className="col-span-full">
                <div className={`p-4 rounded-lg ${isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    <p className="text-center font-medium">
                        {isConnected ? 'Device Connected' : 'Device Not Connected'}
                    </p>
                    {error && (
                        <p className="text-center text-sm mt-2">
                            {error}
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
};

export default WaterQualityMetrics;