import React from 'react';
import { Droplet } from 'lucide-react';

const DeviceSummary = ({ deviceData }) => {
    return (
        <div className="bg-white rounded-lg shadow-md p-6 col-span-1">
            <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mr-4">
                    <Droplet className="h-6 w-6 text-water-medium" />
                </div>
                <div>
                    <h3 className="text-lg font-semibold">{deviceData.deviceId}</h3>
                    <p className="text-gray-500">{deviceData.location}</p>
                </div>
            </div>

            <div className="mt-6">
                <div className="flex justify-between mb-2">
                    <span className="text-gray-600">Status</span>
                    <span className={`font-medium ${deviceData.status === 'active' ? 'text-green-600' :
                        deviceData.status === 'inactive' ? 'text-gray-600' : 'text-orange-600'
                        }`}>
                        {deviceData.status.charAt(0).toUpperCase() + deviceData.status.slice(1)}
                    </span>
                </div>

                <div className="mt-4">
                    <div className="flex justify-between mb-1">
                        <span className="text-gray-600">Water Level</span>
                        <span className="text-gray-800">{deviceData.waterLevel}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div
                            className="bg-water-medium h-2.5 rounded-full"
                            style={{ width: `${deviceData.waterLevel}%` }}
                        ></div>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                        {deviceData.waterLevel} of {deviceData.reservoirCapacity} liters
                    </p>
                </div>
            </div>
        </div>
    );
};

export default DeviceSummary; 