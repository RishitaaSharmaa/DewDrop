import React from 'react';
import { Link } from 'react-router-dom';
import Layout from '../components/Layout';
import { Droplet, Wind, Thermometer, Activity, BarChart2, Shield, Globe, Settings } from 'lucide-react';

const HomePage = () => {
    return (
        <Layout>
            <div className="min-h-screen flex flex-col">
                {/* Hero Section */}
                <section className="py-20 bg-gradient-to-r from-blue-50 to-cyan-50">
                    <div className="container mx-auto px-4">
                        <div className="flex flex-col lg:flex-row items-center gap-12">
                            <div className="lg:w-1/2">
                                <h1 className="text-4xl md:text-5xl font-bold text-water-dark mb-6 leading-tight">
                                    Generate Clean Water <span className="text-water-medium">From Air</span>
                                </h1>
                                <p className="text-xl text-gray-700 mb-8">
                                    Dew<span className='text-water-dark'>D</span>rop is revolutionizing water access with innovative atmospheric water generation technology.
                                    Our devices extract clean drinking water from the air around you, eliminating the need for plastic bottles and water delivery services.
                                </p>
                                <div className="flex flex-col sm:flex-row gap-4">
                                    <Link to="/user-dashboard" className="bg-water-dark hover:bg-water-dark/90 text-white px-6 py-3 rounded-md font-medium">
                                        Get Started
                                    </Link>
                                    <Link to="/device/device-001" className="border border-water-medium text-water-dark px-6 py-3 rounded-md font-medium">
                                        View Your Device
                                    </Link>
                                </div>
                            </div>
                            <div className="lg:w-1/2 flex justify-center">
                                <div className="relative water-drop-shadow">
                                    <div className="w-72 h-72 rounded-full bg-gradient-to-br from-blue-400 to-cyan-500 opacity-20 absolute"></div>
                                    <div className="relative z-10 p-6">
                                        <Droplet className="w-64 h-64 text-water-medium" />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* How It Works Section */}
                <section id="how-it-works" className="py-20 bg-white">
                    <div className="container mx-auto px-4">
                        <div className="text-center mb-16">
                            <h2 className="text-3xl font-bold text-water-dark mb-4">How DewDrop Works</h2>
                            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                                Our advanced high-voltage technology efficiently captures water from desert fog, providing a sustainable water source for arid regions.
                            </p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                            <div className="bg-gradient-to-b from-blue-50 to-white rounded-xl p-8 shadow-sm hover:shadow-md transition-all">
                                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
                                    <Wind className="h-8 w-8 text-water-medium" />
                                </div>
                                <h3 className="text-xl font-semibold text-center mb-4">Fog Detection</h3>
                                <p className="text-gray-600 text-center">
                                    Smart sensors detect optimal fog conditions during early morning hours in desert environments.
                                </p>
                            </div>

                            <div className="bg-gradient-to-b from-blue-50 to-white rounded-xl p-8 shadow-sm hover:shadow-md transition-all">
                                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
                                    <Activity className="h-8 w-8 text-water-medium" />
                                </div>
                                <h3 className="text-xl font-semibold text-center mb-4">High-Voltage Collection</h3>
                                <p className="text-gray-600 text-center">
                                    20,000V electrical field efficiently attracts and captures water molecules from fog particles.
                                </p>
                            </div>

                            <div className="bg-gradient-to-b from-blue-50 to-white rounded-xl p-8 shadow-sm hover:shadow-md transition-all">
                                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
                                    <Droplet className="h-8 w-8 text-water-medium" />
                                </div>
                                <h3 className="text-xl font-semibold text-center mb-4">Collection & Filtration</h3>
                                <p className="text-gray-600 text-center">
                                    Captured water is collected and purified through an advanced filtration system for safe consumption.
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Features Section */}
                <section id="features" className="py-20 bg-gradient-to-r from-blue-50 to-cyan-50">
                    <div className="container mx-auto px-4">
                        <div className="text-center mb-16">
                            <h2 className="text-3xl font-bold text-water-dark mb-4">Smart Water Generation</h2>
                            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                                DewDrop devices are packed with advanced features to provide the best water generation experience.
                            </p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                            <div className="bg-white p-6 rounded-lg shadow-md text-center hover:shadow-lg transition-all">
                                <Activity className="h-10 w-10 text-water-medium mx-auto mb-4" />
                                <h3 className="text-lg font-semibold mb-2">Energy Efficient</h3>
                                <p className="text-gray-600">Uses minimal energy to extract maximum water from the air.</p>
                            </div>

                            <div className="bg-white p-6 rounded-lg shadow-md text-center hover:shadow-lg transition-all">
                                <Shield className="h-10 w-10 text-water-medium mx-auto mb-4" />
                                <h3 className="text-lg font-semibold mb-2">Clean Water</h3>
                                <p className="text-gray-600">Advanced filtration ensures pure, safe drinking water.</p>
                            </div>

                            <div className="bg-white p-6 rounded-lg shadow-md text-center hover:shadow-lg transition-all">
                                <BarChart2 className="h-10 w-10 text-water-medium mx-auto mb-4" />
                                <h3 className="text-lg font-semibold mb-2">Smart Monitoring</h3>
                                <p className="text-gray-600">Track water quality, usage, and device performance from your phone.</p>
                            </div>

                            <div className="bg-white p-6 rounded-lg shadow-md text-center hover:shadow-lg transition-all">
                                <Globe className="h-10 w-10 text-water-medium mx-auto mb-4" />
                                <h3 className="text-lg font-semibold mb-2">Eco-Friendly</h3>
                                <p className="text-gray-600">Reduce plastic waste and your carbon footprint.</p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Device Setup Section */}
                <section id="device-setup" className="py-20 bg-white">
                    <div className="container mx-auto px-4">
                        <div className="text-center mb-16">
                            <h2 className="text-3xl font-bold text-water-dark mb-4">Installing Your Harvesting Station</h2>
                            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                                Setting up your high-voltage fog harvesting station requires professional installation for optimal performance.
                            </p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
                            <div className="text-center">
                                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6 relative">
                                    <span className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-water-dark text-white flex items-center justify-center font-bold">1</span>
                                    <Settings className="h-8 w-8 text-water-medium" />
                                </div>
                                <h3 className="text-xl font-semibold mb-4">Site Assessment</h3>
                                <p className="text-gray-600">
                                    Our team analyzes fog patterns and environmental conditions to determine optimal installation location.
                                </p>
                            </div>

                            <div className="text-center">
                                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6 relative">
                                    <span className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-water-dark text-white flex items-center justify-center font-bold">2</span>
                                    <Activity className="h-8 w-8 text-water-medium" />
                                </div>
                                <h3 className="text-xl font-semibold mb-4">Professional Setup</h3>
                                <p className="text-gray-600">
                                    Expert installation of the 20,000V system with proper safety measures and grounding systems.
                                </p>
                            </div>

                            <div className="text-center">
                                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6 relative">
                                    <span className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-water-dark text-white flex items-center justify-center font-bold">3</span>
                                    <Droplet className="h-8 w-8 text-water-medium" />
                                </div>
                                <h3 className="text-xl font-semibold mb-4">System Activation</h3>
                                <p className="text-gray-600">
                                    Remote monitoring setup and initial system calibration for your specific environment.
                                </p>
                            </div>
                        </div>

                        <div className="text-center mt-12">
                            <Link to="/user-dashboard" className="bg-water-dark hover:bg-water-dark/90 text-white px-6 py-3 rounded-md font-medium">
                                Monitor Your Station
                            </Link>
                        </div>
                    </div>
                </section>
            </div>
        </Layout>
    );
};

export default HomePage;