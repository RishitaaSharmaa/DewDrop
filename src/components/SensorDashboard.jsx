import React, { useState, useEffect } from 'react';
import waterQualityMetrics from '../data/waterQualityMetrics';

const SensorDashboard = () => {
    const [metrics, setMetrics] = useState(waterQualityMetrics.getMetrics());
    const [isConnected, setIsConnected] = useState(false);
    const [ws, setWs] = useState(null);

    useEffect(() => {
        // Subscribe to water quality metrics updates
        const handleMetricsUpdate = (newMetrics) => {
            setMetrics(newMetrics);
        };
        waterQualityMetrics.subscribe(handleMetricsUpdate);

        // WebSocket connection
        const websocket = new WebSocket('ws://localhost:8080');

        websocket.onopen = () => {
            console.log('Connected to WebSocket server');
        };

        websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);

            if (message.type === 'data') {
                waterQualityMetrics.updateMetrics(message.data);
            } else if (message.type === 'connection') {
                setIsConnected(message.status);
            }
        };

        websocket.onclose = () => {
            console.log('Disconnected from WebSocket server');
            setIsConnected(false);
        };

        setWs(websocket);

        return () => {
            websocket.close();
            waterQualityMetrics.unsubscribe(handleMetricsUpdate);
        };
    }, []);

    const formatMetric = (value) => {
        if (value === null) return '--';
        return value;
    };

    return (
        <div className="p-4">
            <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-2xl font-bold mb-4">Water Quality Dashboard</h2>

                <div className="mb-4">
                    <span className={`inline-block w-3 h-3 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
                    <span className="text-gray-700">
                        {isConnected ? 'Device Connected' : 'Device Disconnected'}
                    </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="bg-gray-100 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Temperature</h3>
                        <div className="text-2xl font-mono">
                            {formatMetric(metrics.temperature)} °C
                        </div>
                    </div>

                    <div className="bg-gray-100 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">pH Level</h3>
                        <div className="text-2xl font-mono">
                            {formatMetric(metrics.pH)}
                        </div>
                    </div>

                    <div className="bg-gray-100 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Turbidity</h3>
                        <div className="text-2xl font-mono">
                            {formatMetric(metrics.turbidity)} NTU
                        </div>
                    </div>

                    <div className="bg-gray-100 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Dissolved Oxygen</h3>
                        <div className="text-2xl font-mono">
                            {formatMetric(metrics.dissolvedOxygen)} mg/L
                        </div>
                    </div>

                    <div className="bg-gray-100 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Conductivity</h3>
                        <div className="text-2xl font-mono">
                            {formatMetric(metrics.conductivity)} µS/cm
                        </div>
                    </div>

                    <div className="bg-gray-100 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Last Updated</h3>
                        <div className="text-lg font-mono">
                            {metrics.timestamp ? new Date(metrics.timestamp).toLocaleString() : '--'}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SensorDashboard; 