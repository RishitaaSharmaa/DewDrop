import React from 'react';
import { Droplet } from 'lucide-react';

const LoadingState = () => {
  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col items-center justify-center">
        <div className="animate-pulse">
          <Droplet className="h-16 w-16 text-water-medium" />
        </div>
        <h2 className="mt-6 text-xl font-semibold text-gray-700">Loading Device Data</h2>
        <p className="mt-2 text-gray-500">Please wait while we fetch the latest information...</p>
      </div>
    </div>
  );
};

export default LoadingState; 