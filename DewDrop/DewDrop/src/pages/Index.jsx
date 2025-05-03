import React from 'react';
import { Link } from 'react-router-dom';
import Layout from '../components/Layout';
import { Droplet, Settings, Activity, BarChart2, Shield } from 'lucide-react';

// Simple icon components
const WaterDropIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
    </svg>
);

const AlertIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
);

const ChartIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
);

const DeviceIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
    </svg>
);

const SaveIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z" />
    </svg>
);

const EcoIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);

const Index = () => {
    return (
        <Layout>
            <div className="min-h-screen background-gradient">
                {/* Hero Section */}
                <section className="relative overflow-hidden py-20">
                    <div className="absolute inset-0 z-0 bg-cover bg-center opacity-10"></div>
                    <div className="absolute top-20 right-20 w-72 h-72 bg-blue-400 rounded-full opacity-20 blur-3xl"></div>
                    <div className="absolute bottom-10 left-10 w-64 h-64 bg-cyan-300 rounded-full opacity-20 blur-3xl"></div>

                    <div className="container relative z-10 mx-auto px-4">
                        <div className="flex flex-col items-center text-center max-w-3xl mx-auto">
                            <div className="mb-8 animate-droplet-bounce">
                                <Droplet className="h-20 w-20 text-water-medium" />
                            </div>
                            <h1 className="text-6xl font-bold text-water-dark mb-6">
                                Dew<span className="text-cyan-400">Drop</span>
                            </h1>
                            <p className="text-xl text-gray-700 mb-10">
                                Revolutionary atmospheric water generation technology that creates clean drinking water from air.
                                Monitor your devices, track water quality, and contribute to a sustainable future.
                            </p>

                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 w-full max-w-4xl mx-auto mt-6">
                                <Link to="/home" className="group">
                                    <div className="bg-white/80 backdrop-blur p-6 rounded-xl shadow-lg border border-blue-100 hover:border-water-medium transition-all hover:shadow-xl hover:scale-105 h-full flex flex-col items-center">
                                        <Droplet className="h-10 w-10 text-water-medium mb-3 group-hover:text-water-dark transition-colors" />
                                        <h3 className="font-semibold text-water-dark mb-2">Home</h3>
                                        <p className="text-sm text-gray-600">Learn about our technology</p>
                                    </div>
                                </Link>

                                <Link to="/user-dashboard" className="group">
                                    <div className="bg-white/80 backdrop-blur p-6 rounded-xl shadow-lg border border-blue-100 hover:border-water-medium transition-all hover:shadow-xl hover:scale-105 h-full flex flex-col items-center">
                                        <BarChart2 className="h-10 w-10 text-water-medium mb-3 group-hover:text-water-dark transition-colors" />
                                        <h3 className="font-semibold text-water-dark mb-2">Dashboard</h3>
                                        <p className="text-sm text-gray-600">View your water stats</p>
                                    </div>
                                </Link>

                                <Link to="/device/device-001" className="group">
                                    <div className="bg-white/80 backdrop-blur p-6 rounded-xl shadow-lg border border-blue-100 hover:border-water-medium transition-all hover:shadow-xl hover:scale-105 h-full flex flex-col items-center">
                                        <Activity className="h-10 w-10 text-water-medium mb-3 group-hover:text-water-dark transition-colors" />
                                        <h3 className="font-semibold text-water-dark mb-2">My Device</h3>
                                        <p className="text-sm text-gray-600">Monitor your device</p>
                                    </div>
                                </Link>

                                <Link to="/admin-panel" className="group">
                                    <div className="bg-white/80 backdrop-blur p-6 rounded-xl shadow-lg border border-blue-100 hover:border-water-medium transition-all hover:shadow-xl hover:scale-105 h-full flex flex-col items-center">
                                        <Settings className="h-10 w-10 text-water-medium mb-3 group-hover:text-water-dark transition-colors" />
                                        <h3 className="font-semibold text-water-dark mb-2">Admin</h3>
                                        <p className="text-sm text-gray-600">Manage all devices</p>
                                    </div>
                                </Link>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Features Quick Overview */}
                <section className="py-16 bg-white bg-why">
                    <div className="container mx-auto px-4">
                        <h2 className="text-3xl font-bold text-center text-water-dark mb-16">Why Choose Dew<span className='text-black'>D</span>rop?</h2>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-10 max-w-5xl mx-auto">
                            <div className="text-center">
                                <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mx-auto mb-6">
                                    <Droplet className="h-8 w-8 text-water-medium" />
                                </div>
                                <h3 className="text-xl font-semibold mb-4">Clean Water Anywhere</h3>
                                <p className="text-gray-600">
                                    Generate fresh, clean drinking water from the air around you, even in remote locations.
                                </p>
                            </div>

                            <div className="text-center">
                                <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mx-auto mb-6">
                                    <Shield className="h-8 w-8 text-water-medium" />
                                </div>
                                <h3 className="text-xl font-semibold mb-4">Superior Filtration</h3>
                                <p className="text-gray-600">
                                    Advanced multi-stage filtration ensures water is pure and safe for consumption.
                                </p>
                            </div>

                            <div className="text-center">
                                <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mx-auto mb-6">
                                    <Activity className="h-8 w-8 text-water-medium" />
                                </div>
                                <h3 className="text-xl font-semibold mb-4">Smart Monitoring</h3>
                                <p className="text-gray-600">
                                    Track water quality, usage statistics, and device performance from anywhere.
                                </p>
                            </div>
                        </div>

                        <div className="text-center mt-16">
                            <Link to="/home" className="bg-water-dark hover:bg-water-dark/90 text-white px-6 py-3 rounded-md font-medium text-lg">
                                Learn More
                            </Link>
                        </div>
                    </div>
                </section>
            </div>
        </Layout>
    );
};

// Feature icons and data
const features = [
    {
        icon: <WaterDropIcon />,
        title: "Real-time Monitoring",
        description: "Monitor water usage in real-time with accurate sensors and smart analytics."
    },
    {
        icon: <AlertIcon />,
        title: "Leak Detection",
        description: "Receive instant alerts when unusual water flow patterns are detected."
    },
    {
        icon: <ChartIcon />,
        title: "Usage Analytics",
        description: "Visualize water consumption patterns and identify opportunities for conservation."
    },
    {
        icon: <DeviceIcon />,
        title: "Smart Devices",
        description: "Connect with IoT devices for automated water management and control."
    },
    {
        icon: <SaveIcon />,
        title: "Cost Savings",
        description: "Reduce water bills through efficient usage and waste prevention."
    },
    {
        icon: <EcoIcon />,
        title: "Eco-friendly",
        description: "Contribute to environmental sustainability by conserving water resources."
    }
];

export default Index; 