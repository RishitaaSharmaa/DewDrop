import React from 'react';
import { BarChart2, Zap } from 'lucide-react';

const PerformanceCharts = ({ flowRate, energyConsumption }) => {
    // Mock data for charts
    const weeklyData = [
        { day: 'Mon', value: flowRate * 0.9 },
        { day: 'Tue', value: flowRate * 1.1 },
        { day: 'Wed', value: flowRate * 0.8 },
        { day: 'Thu', value: flowRate * 1.2 },
        { day: 'Fri', value: flowRate * 1.0 },
        { day: 'Sat', value: flowRate * 0.7 },
        { day: 'Sun', value: flowRate * 0.5 },
    ];

    const energyData = [
        { day: 'Mon', value: energyConsumption * 0.95 },
        { day: 'Tue', value: energyConsumption * 1.05 },
        { day: 'Wed', value: energyConsumption * 0.9 },
        { day: 'Thu', value: energyConsumption * 1.1 },
        { day: 'Fri', value: energyConsumption * 1.0 },
        { day: 'Sat', value: energyConsumption * 0.8 },
        { day: 'Sun', value: energyConsumption * 0.7 },
    ];

    // Find max value for scaling
    const maxFlowValue = Math.max(...weeklyData.map(item => item.value));
    const maxEnergyValue = Math.max(...energyData.map(item => item.value));

    return (
        <div className="bg-white rounded-lg shadow-md p-6 col-span-1">
            <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>

            <div className="space-y-8">
                <div>
                    <div className="flex items-center mb-3">
                        <BarChart2 className="h-5 w-5 text-water-medium mr-2" />
                        <h4 className="font-medium">Water Flow Rate</h4>
                        <span className="ml-auto text-lg font-semibold">{flowRate} <span className="text-sm font-normal text-gray-500">L/hr</span></span>
                    </div>

                    <div className="h-24 flex items-end space-x-1">
                        {weeklyData.map((item, index) => (
                            <div key={index} className="flex-1 flex flex-col items-center">
                                <div
                                    className="w-full bg-blue-100 rounded-t"
                                    style={{
                                        height: `${(item.value / maxFlowValue) * 100}%`,
                                        minHeight: '4px'
                                    }}
                                ></div>
                                <span className="text-xs text-gray-500 mt-1">{item.day}</span>
                            </div>
                        ))}
                    </div>
                    <p className="text-xs text-gray-500 mt-2 text-center">Weekly water production (L/hr)</p>
                </div>

                <div>
                    <div className="flex items-center mb-3">
                        <Zap className="h-5 w-5 text-yellow-500 mr-2" />
                        <h4 className="font-medium">Energy Consumption</h4>
                        <span className="ml-auto text-lg font-semibold">{energyConsumption} <span className="text-sm font-normal text-gray-500">kWh</span></span>
                    </div>

                    <div className="h-24 flex items-end space-x-1">
                        {energyData.map((item, index) => (
                            <div key={index} className="flex-1 flex flex-col items-center">
                                <div
                                    className="w-full bg-yellow-100 rounded-t"
                                    style={{
                                        height: `${(item.value / maxEnergyValue) * 100}%`,
                                        minHeight: '4px'
                                    }}
                                ></div>
                                <span className="text-xs text-gray-500 mt-1">{item.day}</span>
                            </div>
                        ))}
                    </div>
                    <p className="text-xs text-gray-500 mt-2 text-center">Weekly energy usage (kWh)</p>
                </div>
            </div>
        </div>
    );
};

export default PerformanceCharts; 