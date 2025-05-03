import React from 'react';
import { Link } from 'react-router-dom';
import Layout from '../components/Layout';

const NotFound = () => {
    return (
        <Layout>
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-6xl font-bold text-water-dark mb-4">404</h1>
                    <h2 className="text-2xl font-medium text-gray-700 mb-6">Page Not Found</h2>
                    <p className="text-gray-600 mb-8">The page you are looking for might have been removed or is temporarily unavailable.</p>
                    <Link
                        to="/"
                        className="px-4 py-2 bg-water-medium text-white rounded-md hover:bg-water-dark transition-colors"
                    >
                        Return Home
                    </Link>
                </div>
            </div>
        </Layout>
    );
};

export default NotFound; 