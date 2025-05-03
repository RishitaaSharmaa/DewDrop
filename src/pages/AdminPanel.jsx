import React from 'react';
import Layout from '../components/Layout';
import { Droplet, Users, Activity, Settings, Edit, Trash2 } from 'lucide-react';
import AdminMap from '../components/admin/AdminMap';

const AdminPanel = () => {
    // Mock data
    const devices = [
        { id: 'FHS001', name: 'Desert Station Alpha', location: 'Dune Valley', status: 'Active', voltage: '20,000V', fogDensity: 'High', lastMaintenance: '2024-03-15' },
        { id: 'FHS002', name: 'Mountain Ridge Beta', location: 'Highland Desert', status: 'Inactive', voltage: '19,800V', fogDensity: 'Low', lastMaintenance: '2024-03-10' },
        { id: 'FHS003', name: 'Coastal Station Gamma', location: 'Desert Coast', status: 'Maintenance', voltage: '20,000V', fogDensity: 'Medium', lastMaintenance: '2024-03-01' },
    ];

    const stats = {
        totalStations: 15,
        activeStations: 12,
        servedCommunities: 8,
        waterHarvested: 15000
    };

    return (
        <Layout>
            <div className="container mx-auto px-4 py-8">
                <div className="flex justify-between items-center mb-8">
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900">Fog Harvesting Control Center</h1>
                        <p className="text-gray-500">Manage your desert water harvesting network</p>
                    </div>
                    <button className="px-4 py-2 bg-water-medium text-white rounded-md hover:bg-water-dark transition-colors">
                        Add New Station
                    </button>
                </div>

                {/* Stats Overview */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                    <div className="bg-white rounded-lg shadow p-6">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-sm font-medium">Total Stations</h3>
                            <Droplet className="h-4 w-4 text-water-dark" />
                        </div>
                        <div className="text-2xl font-bold">{stats.totalStations}</div>
                        <p className="text-xs text-gray-500">Active harvesting points</p>
                    </div>

                    <div className="bg-white rounded-lg shadow p-6">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-sm font-medium">Operational Status</h3>
                            <Activity className="h-4 w-4 text-green-500" />
                        </div>
                        <div className="text-2xl font-bold">{stats.activeStations}</div>
                        <p className="text-xs text-gray-500">Currently collecting</p>
                    </div>

                    <div className="bg-white rounded-lg shadow p-6">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-sm font-medium">Communities Served</h3>
                            <Users className="h-4 w-4 text-blue-500" />
                        </div>
                        <div className="text-2xl font-bold">{stats.servedCommunities}</div>
                        <p className="text-xs text-gray-500">Desert regions supported</p>
                    </div>

                    <div className="bg-white rounded-lg shadow p-6">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-sm font-medium">Water Harvested</h3>
                            <Droplet className="h-4 w-4 text-water-dark" />
                        </div>
                        <div className="text-2xl font-bold">{stats.waterHarvested}L</div>
                        <p className="text-xs text-gray-500">Total fog collection</p>
                    </div>
                </div>

                {/* Station Map */}
                <div className="bg-white rounded-lg shadow mb-8">
                    <div className="px-6 py-4 border-b border-gray-200">
                        <h2 className="text-lg font-semibold text-gray-900">Station Locations</h2>
                        <p className="text-sm text-gray-500">Fog harvesting stations in Rajasthan desert region</p>
                    </div>
                    <div className="p-6">
                        <div className="relative h-[400px] w-full rounded-lg overflow-hidden">
                            <iframe
                                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d458798.33851090485!2d72.81517083403321!3d24.63197699921499!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3942d8c6a4021605%3A0x43d7c02fc0839b4c!2sThar%20Desert!5e0!3m2!1sen!2sin!4v1709667947959!5m2!1sen!2sin"
                                className="absolute inset-0 w-full h-full"
                                style={{ border: 0 }}
                                allowFullScreen=""
                                loading="lazy"
                                referrerPolicy="no-referrer-when-downgrade"
                            ></iframe>
                            {/* Station Markers */}
                            {devices.map((device) => (
                                <div
                                    key={device.id}
                                    className={`absolute w-3 h-3 rounded-full transform -translate-x-1/2 -translate-y-1/2 cursor-pointer
                                        ${device.status === 'Active' ? 'bg-green-500' : 
                                          device.status === 'Inactive' ? 'bg-gray-500' : 
                                          'bg-yellow-500'}`}
                                    style={{
                                        // Random positions for demonstration
                                        left: `${30 + Math.random() * 40}%`,
                                        top: `${30 + Math.random() * 40}%`
                                    }}
                                    title={`${device.name} - ${device.status}`}
                                >
                                    <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-white px-2 py-1 rounded shadow text-xs whitespace-nowrap">
                                        {device.id}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Devices Table */}
                <div className="bg-white rounded-lg shadow overflow-hidden">
                    <div className="px-6 py-4 border-b border-gray-200">
                        <h2 className="text-lg font-semibold text-gray-900">Managed Stations</h2>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Station ID</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Location</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Voltage</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Fog Density</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Last Maintenance</th>
                                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {devices.map((device) => (
                                    <tr key={device.id}>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{device.id}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{device.name}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{device.location}</td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                                                device.status === 'Active' ? 'bg-green-100 text-green-800' :
                                                device.status === 'Inactive' ? 'bg-gray-100 text-gray-800' :
                                                'bg-yellow-100 text-yellow-800'
                                            }`}>
                                                {device.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{device.voltage}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{device.fogDensity}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{device.lastMaintenance}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                            <button className="text-water-medium hover:text-water-dark mr-3" title="Edit Station">
                                                <Edit className="h-4 w-4 inline" />
                                            </button>
                                            <button className="text-red-500 hover:text-red-700" title="Delete Station">
                                                <Trash2 className="h-4 w-4 inline" />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </Layout>
    );
};

export default AdminPanel;