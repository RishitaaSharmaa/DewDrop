import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Droplet } from 'lucide-react';
import '../styles/theme.css';
import { AnimatedTooltip } from './ui/animated-tooltip';
import { Toaster } from './ui/toast';

const Layout = ({ children }) => {
    const location = useLocation();

    // Add the isActive function
    const isActive = (path) => {
        return location.pathname === path;
    };

    return (
        <div className="min-h-screen flex flex-col">
            <header className="bg-white/80 backdrop-blur-sm shadow-sm">
                <div className="container mx-auto px-4 py-4">
                    <div className="flex justify-between items-center">
                        <Link to="/" className="flex items-center space-x-2">
                            <Droplet className="h-8 w-8 text-black" />
                            <span className="text-2xl font-bold text-black">Dew<span className='text-water-dark text-3xl'>D</span>rop</span>
                        </Link>
                        <nav className="flex space-x-6">
                            <Link
                                to="/"
                                className={`text-sm font-medium hover:text-black transition-colors ${isActive('/') ? 'text-black font-bold border-b-2 border-water-dark pb-1' : 'text-gray-600'}`}
                            >
                                Home
                            </Link>
                            <Link
                                to="/home"
                                className={`text-sm font-medium hover:text-black transition-colors ${isActive('/home') ? 'text-black border-b-2 border-water-dark pb-1' : 'text-gray-600'}`}
                            >
                                Features
                            </Link>
                            <Link
                                to="/device/device-001"
                                className={`text-sm font-medium hover:text-black transition-colors ${location.pathname.includes('/device') ? 'text-black border-b-2 border-water-dark pb-1' : 'text-gray-600'}`}
                            >
                                My Device
                            </Link>
                            <Link
                                to="/user-dashboard"
                                className={`text-sm font-medium hover:text-black transition-colors ${isActive('/user-dashboard') ? 'text-black border-b-2 border-water-dark pb-1' : 'text-gray-600'}`}
                            >
                                User Dashboard
                            </Link>
                            <Link
                                to="/admin-panel"
                                className={`text-sm font-medium hover:text-black transition-colors ${isActive('/admin-panel') ? 'text-black border-b-2 border-water-dark pb-1' : 'text-gray-600'}`}
                            >
                                Admin Panel
                            </Link>
                        </nav>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1">
                {children}
            </main>

            {/* Footer */}
            <footer className="bg-white/80 backdrop-blur-sm bg-footer">
                <div className="container mx-auto px-4 py-6">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                            <Droplet className="h-5 w-5 text-water-dark" />
                            <span className="text-sm text-gray-600">Dew<span className='text-water-dark'>D</span>rop</span>
                        </div>
                        <p className="text-sm text-gray-500">© 2025 Dew<span className='text-water-dark'>D</span>rop. All rights reserved.</p>
                    </div>
                </div>
            </footer>
            
            {/* Toast notifications */}
            <Toaster />
        </div>
    );
};

export default Layout;