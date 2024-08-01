import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { RingLoader } from 'react-spinners'; // Import loading animation
import { Doughnut } from 'react-chartjs-2'; // Import Doughnut chart
import 'chart.js/auto'; // Import Chart.js
import ChartDataLabels from 'chartjs-plugin-datalabels'; // Import Chart.js Data Labels plugin
import './Dashboard.css';
// import './index.css'; // Import Tailwind CSS styles

const Dashboard = () => {
    const [emails, setEmails] = useState([]);
    const [totalEmails, setTotalEmails] = useState(0);
    const [flaggedCount, setFlaggedCount] = useState(0);
    const [loading, setLoading] = useState(true); // State for loading animation
    const url = 'http://localhost:5000/fetch_emails';

    useEffect(() => {
        axios.get(url)
            .then(response => {
                console.log(response.data); // Log the response for debugging
                setEmails(response.data.all_emails);
                setTotalEmails(response.data.total_emails);
                setFlaggedCount(response.data.flagged_count);
                setLoading(false); // Turn off loading animation
                console.log(response.data.all_emails)
            })
            .catch(error => {
                console.error('Error fetching emails:', error);
                setLoading(false); // Turn off loading animation even in case of error
            });
    }, []);

    const doughnutData = {
        labels: ['Other Emails', 'Flagged Emails'], // Change order of labels
        datasets: [
            {
                label: 'Emails',
                data: [totalEmails - flaggedCount, flaggedCount], // Change order of data
                backgroundColor: ['#36A2EB', '#FF6384'], // Update colors to match the new order
                hoverBackgroundColor: ['#36A2EB', '#FF6384']
            }
        ]
    };

    const doughnutOptions = {
        plugins: {
            datalabels: {
                color: '#fff',
                formatter: (value, context) => {
                    const total = context.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
                    const percentage = (value / total * 100).toFixed(2) + '%';
                    return percentage;
                }
            },
            tooltip: {
                callbacks: {
                    label: (tooltipItem) => {
                        const dataset = tooltipItem.dataset;
                        const dataIndex = tooltipItem.dataIndex;
                        const value = dataset.data[dataIndex];
                        const total = dataset.data.reduce((a, b) => a + b, 0);
                        const percentage = (value / total * 100).toFixed(2) + '%';
                        return `${tooltipItem.label}: ${value} (${percentage})`;
                    }
                }
            },
            doughnutlabel: {
                labels: [
                    {
                        text: `${((flaggedCount / totalEmails) * 100).toFixed(2)}%`,
                        font: {
                            size: '20'
                        },
                        color: '#FF6384'
                    }
                ]
            }
        },
        maintainAspectRatio: false,
        responsive: true,
        legend: {
            position: 'top',
        }
    };    

    return (
        <div className="dashboard">
            <h1 className="text-4xl mb-4 text-muted">Phishing Detection Dashboard</h1>
            {loading ? (
                <>
                    <div className="emails">
                        <div className='infoMsg'>
                            <strong>Loading Mails...</strong>
                        </div>
                        <div className="flex justify-center items-center">
                            <RingLoader color="#3B82F6" loading={loading} size={80} /> {/* Loading animation */}
                        </div>
                    </div>
                </>
            ) : (
                <>
                    <div className="stats mb-8">
                        <div className="stat-item-total">
                            <div className="stat-title">Total Emails</div>
                            <div className="stat-value">{totalEmails}</div>
                        </div>
                        <div className="stat-item-flagged">
                            <div className="stat-title">Flagged Emails</div>
                            <div className="stat-value">{flaggedCount}</div>
                        </div>
                    </div>
                    <div className="chart-container mb-8" style={{ height: '300px' }}>
                        <Doughnut data={doughnutData} options={doughnutOptions} plugins={[ChartDataLabels]} />
                    </div>
                    <div className="emails">
                        {emails.map((email, index) => (
                            <div key={index} className={`email ${email.is_phishing ? 'flagged' : ''}`}>
                                <h3>{email.subject}</h3>
                                <p>{email.body}</p>
                                <p>Status: {email.is_phishing ? 'Phishing' : 'Not Phishing'}</p>
                            </div>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
};

export default Dashboard;
