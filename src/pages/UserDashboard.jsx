import React from 'react';
import { Link } from 'react-router-dom';
import Layout from '../components/Layout';
import { Droplet, Thermometer } from 'lucide-react';
import WaterUsageChart from '../components/charts/WaterUsageChart';

const UserDashboard = () => {
    // Mock data for water usage history
    const waterUsageData = [
        { date: '2024-01-01', generated: 15, consumed: 12 },
        { date: '2024-01-02', generated: 14, consumed: 10 },
        { date: '2024-01-03', generated: 18, consumed: 15 },
        { date: '2024-01-04', generated: 16, consumed: 13 },
        { date: '2024-01-05', generated: 15, consumed: 12 },
        { date: '2024-01-06', generated: 17, consumed: 14 },
        { date: '2024-01-07', generated: 19, consumed: 16 }
    ];

    // Mock data for devices
    // Update the devices mock data
    const devices = [
        { name: 'Desert Station Alpha', status: 'Online', location: 'Dune Valley' },
        { name: 'Fog Collector Beta', status: 'Offline', location: 'Mountain Ridge' },
        { name: 'Harvester Station Gamma', status: 'Maintenance', location: 'Coastal Desert' }
    ];

    // Mock data for status cards
    // const statusData = {
    //     waterQuality: {
    //         status: 'Excellent',
    //         tdsLevel: '120 ppm',
    //         phLevel: '7.2',
    //         purity: '99%'
    //     },
    //     maintenance: {
    //         status: 'All Good',
    //         nextMaintenance: '45 days',
    //         filterStatus: '85%'
    //     },
    //     environmental: {
    //         bottlesSaved: '631',
    //         co2Reduced: '287 kg',
    //         waterMilesSaved: '1542 miles'
    //     }
    // };

    // Add new mock data for top cards
    const dashboardStats = {
        totalWater: '1254.8 L',
        tankLevel: '75%',
        waterCollection: {
            amount: '5.8 L',
            rate: '2.3 L/hr',
            peakHours: '4AM - 8AM'
        },
        weather: {
            temperature: '18°C',
            humidity: '85%',
            condition: 'Heavy Fog',
            isOptimal: true
        }
    };

    // Mock data for status cards
    const statusData = {
        systemStatus: {
            voltage: '20,000V',
            collectionRate: '2.3 L/hr',
            efficiency: '95%',
            fogDensity: 'High'
        },
        maintenance: {
            status: 'Operational',
            nextMaintenance: '45 days',
            voltageSystem: '98%',
            collectorGrid: '85%'
        },
        environmental: {
            waterGenerated: '1,631 L',
            communitiesServed: '3',
            peopleImpacted: '450+'
        },
        waterQuality: {
            status: 'Excellent',
            tdsLevel: '120 ppm',
            phLevel: '7.2',
            purity: '99%'
        }
    };

    return (
        <Layout>
            <div className="container mx-auto px-4 py-8">
                <div className="mb-8">
                    <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
                </div>

                <div className="mb-8">
                    <div className="flex justify-between items-center mb-6">
                        <div>
                            <h2 className="text-2xl font-bold text-gray-900">My Water Dashboard</h2>
                            <p className="text-gray-500">Monitor and manage your DewDrop devices and water generation</p>
                        </div>
                        <Link
                            to="/device/main"
                            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
                        >
                            View Main Device
                        </Link>
                    </div>

                    {/* Top Stats Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {/* Total Water Generated */}
                        <div className="bg-white rounded-lg shadow p-6">
                            <div className="flex justify-between items-center mb-2">
                                <h3 className="text-sm font-medium text-gray-500">Total Water Generated</h3>
                                <Droplet className="h-5 w-5 text-blue-500" />
                            </div>
                            <div className="mt-2">
                                <div className="text-2xl font-bold">{dashboardStats.totalWater}</div>
                                <p className="text-sm text-gray-500">Your lifetime water generation</p>
                            </div>
                        </div>

                        {/* Water Tank Level */}
                        <div className="bg-white rounded-lg shadow p-6">
                            <div className="flex justify-between items-center mb-2">
                                <h3 className="text-sm font-medium text-gray-500">Water Tank Level</h3>
                                <Droplet className="h-5 w-5 text-blue-500" />
                            </div>
                            <div className="mt-2">
                                <div className="text-2xl font-bold">{dashboardStats.tankLevel}</div>
                                <div className="mt-2">
                                    <div className="h-2 bg-gray-100 rounded-full">
                                        <div
                                            className="h-2 bg-blue-500 rounded-full"
                                            style={{ width: dashboardStats.tankLevel }}
                                        ></div>
                                    </div>
                                </div>
                                <p className="text-sm text-gray-500 mt-1">Remaining: ~3 days</p>
                            </div>
                        </div>

                        {/* Water Collection */}
                        <div className="bg-white rounded-lg shadow p-6">
                            <div className="flex justify-between items-center mb-2">
                                <h3 className="text-sm font-medium text-gray-500">Water Collection</h3>
                                <svg className="h-5 w-5 text-blue-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M12 2v20M2 12h20" />
                                </svg>
                            </div>
                            <div className="mt-2">
                                <div className="text-2xl font-bold">{dashboardStats.waterCollection.amount}</div>
                                <p className="text-sm text-gray-500">Today's collection at {dashboardStats.waterCollection.rate}</p>
                            </div>
                        </div>

                        {/* Weather */}
                        <div className="bg-white rounded-lg shadow p-6">
                            <div className="flex justify-between items-center mb-2">
                                <h3 className="text-sm font-medium text-gray-500">Weather</h3>
                                <Thermometer className="h-5 w-5 text-blue-500" />
                            </div>
                            <div className="mt-2">
                                <div className="text-2xl font-bold">{dashboardStats.weather.temperature}, {dashboardStats.weather.humidity}</div>
                                <p className="text-sm">
                                    <span className="text-gray-500">{dashboardStats.weather.condition} - </span>
                                    <span className="text-green-500">Optimal for collection</span>
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                    {/* Water Usage History */}
                    <div className="lg:col-span-2 bg-white rounded-lg shadow p-6">
                        <h2 className="text-xl font-bold text-gray-900 mb-4">Water Usage History</h2>
                        <div className="h-64">
                            <WaterUsageChart data={waterUsageData} />
                        </div>
                    </div>

                    {/* My Devices section */}
                    <div className="bg-white rounded-lg shadow p-6">
                        <h2 className="text-xl font-bold text-gray-900 mb-4">Harvesting Stations</h2>
                        <div className="space-y-4">
                            {devices.map((device, index) => (
                                <div key={index} className="flex items-center justify-between">
                                    <div>
                                        <h3 className="text-sm font-medium">{device.name}</h3>
                                        <p className="text-xs text-gray-500">{device.location}</p>
                                        <span className={`inline-flex items-center text-xs ${
                                            device.status === 'Online' ? 'text-green-600' :
                                            device.status === 'Offline' ? 'text-red-600' :
                                            'text-yellow-600'
                                        }`}>
                                            <span className={`w-2 h-2 rounded-full mr-1 ${
                                                device.status === 'Online' ? 'bg-green-600' :
                                                device.status === 'Offline' ? 'bg-red-600' :
                                                'bg-yellow-600'
                                            }`}></span>
                                            {device.status}
                                        </span>
                                    </div>
                                    <Link
                                        to="/device/details"
                                        className="text-water-medium hover:text-water-dark text-sm"
                                    >
                                        View
                                    </Link>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Status Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Water Quality */}
                    <div className="bg-white rounded-lg shadow p-6">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-lg font-semibold">Water Quality</h2>
                            <Droplet className="h-5 w-5 text-water-dark" />
                        </div>
                        <div className="text-lg font-bold text-green-600 mb-4">{statusData.waterQuality.status}</div>
                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between text-sm mb-1">
                                    <span>TDS Level</span>
                                    <span>{statusData.waterQuality.tdsLevel}</span>
                                </div>
                                <div className="h-2 bg-gray-100 rounded-full">
                                    <div className="h-2 bg-water-medium rounded-full" style={{ width: '40%' }}></div>
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between text-sm mb-1">
                                    <span>pH Level</span>
                                    <span>{statusData.waterQuality.phLevel}</span>
                                </div>
                                <div className="h-2 bg-gray-100 rounded-full">
                                    <div className="h-2 bg-water-medium rounded-full" style={{ width: '72%' }}></div>
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between text-sm mb-1">
                                    <span>Purity</span>
                                    <span>{statusData.waterQuality.purity}</span>
                                </div>
                                <div className="h-2 bg-gray-100 rounded-full">
                                    <div className="h-2 bg-water-medium rounded-full" style={{ width: '99%' }}></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Maintenance Status */}
                    <div className="bg-white rounded-lg shadow p-6">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-lg font-semibold">Maintenance Status</h2>
                            <svg className="h-5 w-5 text-water-dark" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M19 14a7 7 0 1 1-7-7 7 7 0 0 1 7 7Z" />
                                <path d="M12 7v7l4 4" />
                            </svg>
                        </div>
                        <div className="text-lg font-bold text-green-600 mb-4">{statusData.maintenance.status}</div>
                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between text-sm mb-1">
                                    <span>Next Maintenance</span>
                                    <span>{statusData.maintenance.nextMaintenance}</span>
                                </div>
                                <div className="h-2 bg-gray-100 rounded-full">
                                    <div className="h-2 bg-water-medium rounded-full" style={{ width: '60%' }}></div>
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between text-sm mb-1">
                                    <span>Filter Status</span>
                                    <span>{statusData.maintenance.filterStatus}</span>
                                </div>
                                <div className="h-2 bg-gray-100 rounded-full">
                                    <div className="h-2 bg-water-medium rounded-full" style={{ width: '85%' }}></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Environmental Impact */}
                    <div className="bg-white rounded-lg shadow p-6">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="text-lg font-semibold">Community Impact</h2>
                            <Droplet className="h-5 w-5 text-water-dark" />
                        </div>
                        <div className="space-y-6">
                            <div>
                                <p className="text-sm text-gray-600">Water Generated</p>
                                <p className="text-2xl font-bold">{statusData.environmental.waterGenerated}</p>
                                <p className="text-xs text-gray-500">Total water harvested from fog</p>
                            </div>
                            <div>
                                <p className="text-sm text-gray-600">Communities Served</p>
                                <p className="text-2xl font-bold">{statusData.environmental.communitiesServed}</p>
                                <p className="text-xs text-gray-500">Desert communities with water access</p>
                            </div>
                            <div>
                                <p className="text-sm text-gray-600">People Impacted</p>
                                <p className="text-2xl font-bold">{statusData.environmental.peopleImpacted}</p>
                                <p className="text-xs text-gray-500">Individuals with improved water access</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </Layout>
    );
};

export default UserDashboard;
