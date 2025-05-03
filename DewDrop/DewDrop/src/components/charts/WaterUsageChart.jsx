import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label } from 'recharts';

const WaterUsageChart = ({ data }) => {
    return (
        <ResponsiveContainer width="100%" height="100%">
            <BarChart
                data={data}
                margin={{
                    top: 20,
                    right: 30,
                    left: 20,
                    bottom: 5,
                }}
            >
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis
                    dataKey="date"
                    tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'numeric', day: 'numeric' })}
                    axisLine={false}
                    tickLine={false}
                />
                <YAxis
                    axisLine={false}
                    tickLine={false}
                    ticks={[0, 5, 10, 15, 20]}
                >
                    <Label
                        value="L"
                        position="left"
                        offset={-5}
                        style={{ textAnchor: 'middle', fill: '#666' }}
                    />
                </YAxis>
                <Tooltip
                    formatter={(value, name) => [`${value}L`, name === 'generated' ? 'Generated' : 'Consumed']}
                    labelFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'long', day: 'numeric' })}
                    contentStyle={{
                        backgroundColor: 'white',
                        border: 'none',
                        borderRadius: '8px',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                    }}
                />
                <Legend
                    verticalAlign="top"
                    height={36}
                />
                <Bar
                    dataKey="generated"
                    fill="#60A5FA"
                    name="Generated"
                    radius={[4, 4, 0, 0]}
                />
                <Bar
                    dataKey="consumed"
                    fill="#2563EB"
                    name="Consumed"
                    radius={[4, 4, 0, 0]}
                />
            </BarChart>
        </ResponsiveContainer>
    );
};

export default WaterUsageChart; 