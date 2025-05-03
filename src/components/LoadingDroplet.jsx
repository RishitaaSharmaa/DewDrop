import React from 'react';
import { Droplet } from 'lucide-react';

const LoadingDroplet = () => {
    return (
        <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-full shadow-lg">
                <Droplet
                    className="h-16 w-16 text-water-dark animate-pulse"
                    strokeWidth={1.5}
                    fill="currentColor"
                />
            </div>
        </div>
    );
};

export default LoadingDroplet; 