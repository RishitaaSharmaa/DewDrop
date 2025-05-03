import React from 'react';
import { RefreshCw, Power, Bell, Settings } from 'lucide-react';
import { toast } from '../../components/ui/use-toast.js';  // Changed back to using 'toast'
import axios from 'axios';

const QuickActions = () => {
    const handleAction = (action) => {
        console.log(`Action triggered: ${action}`); // Debugging log
        switch (action) {
            case 'refresh':
                toast({
                    title: "Refreshing device data",
                    description: "Fetching the latest information from your device.",
                });
                break;
            case 'restart':
                toast({
                    title: "Restarting device",
                    description: "Your device will restart in a few moments.",
                });
                break;
            case 'notifications':
                toast({
                    title: "Notification settings",
                    description: "Notification preferences have been updated.",
                });
                break;
            case 'settings':
                toast({
                    title: "Device settings",
                    description: "Opening device configuration panel.",
                });
                break;
            default:
                break;
        }
    };

    const handleStart = async() => {
        try {
            const response = await axios.get('http://localhost:3000/start');
            console.log(response.data);
        } catch (error) {
            console.error('Error:', error);
        }
    }

    const handleStop = async() => {
        try {
            const response = await axios.get('http://localhost:3000/stop');
            console.log(response.data);
        } catch (error) {
            console.error('Error:', error);
        }
    }


    
    return (
        <div className="bg-white rounded-lg shadow-md p-6 col-span-1">
            <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
            <div className="grid grid-cols-2 gap-4">
                <button
                    // onClick={() => handleAction('refresh')}
                    onClick={() => handleRefresh()}
                    className="flex flex-col items-center justify-center p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                    <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center mb-2">
                        <RefreshCw className="h-5 w-5 text-water-medium" />
                    </div>
                    <span className="text-sm font-medium">Refresh Data</span>
                </button>

                <button
                    // onClick={() => handleAction('restart')}
                    onClick={() => handleStart()}
                    className="flex flex-col items-center justify-center p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                    <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center mb-2">
                        <Power className="h-5 w-5 text-green-600" />
                    </div>
                    <span className="text-sm font-medium">Start Device</span>
                </button>

                <button
                    // onClick={() => handleAction('notifications')}
                    onClick={() => handleStop()}
                    className="flex flex-col items-center justify-center p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                    
                    <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center mb-2">
                        <Power className="h-5 w-5 text-red-600" />
                    </div>
                    <span className="text-sm font-medium">Stop</span>
                </button>

                <button
                    onClick={() => handleAction('settings')}
                    className="flex flex-col items-center justify-center p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                    <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center mb-2">
                        <Settings className="h-5 w-5 text-purple-600" />
                    </div>
                    <span className="text-sm font-medium">Settings</span>
                </button>
            </div>
        </div>
    );
};

export default QuickActions;