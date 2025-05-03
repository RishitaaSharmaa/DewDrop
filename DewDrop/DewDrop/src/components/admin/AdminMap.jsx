import React from 'react';
import { MapPin } from 'lucide-react';

const AdminMap = ({ devices }) => {
    return (
        <div className="relative w-full h-[400px] bg-blue-50 overflow-hidden rounded-lg">
            {/* World map background */}
            <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1530569673472-307dc017a82d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1288&q=80')] bg-cover opacity-20"></div>

            {/* Placeholder content */}
            <div className="absolute inset-0 p-8 flex items-center justify-center">
                <div className="text-center text-water-dark">
                    <div className="mb-4">
                        <MapPin className="h-8 w-8 mx-auto text-water-dark" />
                    </div>
                    <p className="text-lg font-medium mb-2">Interactive Map View</p>
                    <p className="text-sm text-gray-600 max-w-md">
                        This area would display an interactive map showing the location and status of all {devices.length} devices across your deployment network.
                    </p>
                </div>
            </div>

            {/* Sample device pins */}
            {devices.map((device, index) => (
                <div
                    key={device.id}
                    className="absolute"
                    style={{
                        top: `${20 + Math.random() * 60}%`,
                        left: `${20 + Math.random() * 60}%`,
                    }}
                >
                    <div className={`
                        flex items-center justify-center h-4 w-4 rounded-full 
                        ${device.status === 'Active' ? 'bg-green-500' :
                            device.status === 'Maintenance' ? 'bg-yellow-500' : 'bg-red-500'} 
                        relative z-10
                    `}>
                        <div className={`
                            absolute inset-0 rounded-full 
                            ${device.status === 'Active' ? 'bg-green-500' :
                                device.status === 'Maintenance' ? 'bg-yellow-500' : 'bg-red-500'} 
                            animate-ping opacity-75
                        `}></div>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default AdminMap; 