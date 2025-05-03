import React from 'react';
import { useParams } from 'react-router-dom';
import Layout from '../components/Layout';
import DeviceSummary from '../components/device/DeviceSummary';
import WaterQualityMetrics from '../components/device/WaterQualityMetrics';
import PerformanceCharts from '../components/device/PerformanceCharts';
import QuickActions from '../components/device/QuickActions';

const DevicePage = () => {
    const { deviceId } = useParams();

    // Mock fog harvesting station data
    const deviceData = {
        deviceId: deviceId,
        name: 'Vihaan 8.0',
        location: 'DTU Campus, Delhi',
        status: 'Active',
        voltage: '20,000V',
        fogDensity: 'High',
        collectionRate: '2.3 L/hr',
        waterCollected: 158.4,
        tankCapacity: 500,
        tankLevel: 75,
        metrics: {
            humidity: '85%',
            temperature: '18°C',
            windSpeed: '12 km/h',
            collectionEfficiency: '95%'
        },
        waterQuality: {
            tds: 120,
            ph: 7.2,
            purity: '99%'
        },
        maintenance: {
            lastCheck: '2025-03-15',
            nextScheduled: '2025-04-15',
            gridCondition: '85%',
            systemHealth: '92%'
        }
    };

    return (
        <Layout>
            <div className="min-h-screen bg-background">
                <div className="container mx-auto px-4 py-8">
                    <h1 className="text-3xl font-bold text-water-dark mb-8">
                        Fog Harvesting Station - {deviceData.name}
                    </h1>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <DeviceSummary deviceData={deviceData} />
                        <QuickActions />
                        <WaterQualityMetrics
                            tds={deviceData.waterQuality.tds}
                            ph={deviceData.waterQuality.ph}
                            purity={deviceData.waterQuality.purity}
                        />
                        <PerformanceCharts
                            collectionRate={deviceData.collectionRate}
                            fogDensity={deviceData.fogDensity}
                            efficiency={deviceData.metrics.collectionEfficiency}
                            voltage={deviceData.voltage}
                        />


                        {/* python models */}
                        <div className="col-span-2 bg-white rounded-lg shadow-lg p-4 min-h-screen">
                            <h2 className="text-xl font-semibold mb-4">Fog Density Analysis</h2>
                            <div className="w-full h-[600px]">
                                <iframe
                                    src="http://localhost:8501"
                                    width="100%"
                                    height="750vh"
                                    title="Python Model Visualization"
                                    className="border-none rounded-md"
                                />
                            </div>
                        </div>

                        <div className="col-span-2 bg-white rounded-lg shadow-lg p-4 min-h-screen">
                            <h2 className="text-xl font-semibold mb-4">Fog Density Analysis</h2>
                            <div className="w-full h-[600px]">
                                <iframe
                                    src="http://localhost:8502"
                                    width="100%"
                                    height="750vh"
                                    title="Python Model Visualization"
                                    className="border-none rounded-md"
                                />
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </Layout>
    );
};

export default DevicePage;