import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { RingLoader } from 'react-spinners';
import { Doughnut, Bar } from 'react-chartjs-2';
import 'chart.js/auto';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import './Dashboard.css';

const Dashboard = () => {
    const [emails, setEmails] = useState([]);
    const [totalEmails, setTotalEmails] = useState(0);
    const [flaggedCount, setFlaggedCount] = useState(0);
    const [emailLengths, setEmailLengths] = useState([]);
    const [loading, setLoading] = useState(true);
    const [phishingKeywords, setPhishingKeywords] = useState([]);  // Add state for keywords

    const [currentPage, setCurrentPage] = useState(1);
    const [emailsPerPage] = useState(30); // Number of emails per page

    const url = 'http://localhost:5000/fetch_emails';

    useEffect(() => {
        axios.get(url)
            .then(response => {
                //console.log(response.data);
                setEmails(response.data.all_emails);
                setTotalEmails(response.data.total_emails);
                setFlaggedCount(response.data.flagged_count);
                setEmailLengths(response.data.email_lengths);
                setPhishingKeywords(response.data.phishing_keywords || []); 
                setLoading(false);
            })
            .catch(error => {
                console.error('Error fetching emails:', error);
                setLoading(false);
            });
    }, []);

    // Pagination logic
    const indexOfLastEmail = currentPage * emailsPerPage;
    const indexOfFirstEmail = indexOfLastEmail - emailsPerPage;
    const currentEmails = emails.slice(indexOfFirstEmail, indexOfLastEmail);

    const paginate = pageNumber => setCurrentPage(pageNumber);

    const highlightKeywords = (text, keywords) => {
        const regex = new RegExp(`\\b(${keywords.join('|')})\\b`, 'gi');
        return text.replace(regex, match => `<span class="highlight">${match}</span>`);
    };

    const doughnutData = {
        labels: ['Other Emails', 'Flagged Emails'],
        datasets: [
            {
                label: 'Emails',
                data: [totalEmails - flaggedCount, flaggedCount],
                backgroundColor: ['#36A2EB', '#FF6384'],
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
                        const total = tooltipItem.dataset.data.reduce((a, b) => a + b, 0);
                        const value = tooltipItem.raw;
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

    const barData = {
        labels: Array.from({ length: emailLengths.length }, (_, i) => i + 1),
        datasets: [
            {
                label: 'Email Lengths',
                data: emailLengths,
                backgroundColor: 'rgba(75,192,192,0.4)',
                borderColor: 'rgba(75,192,192,1)',
                borderWidth: 1
            }
        ]
    };

    const barOptions = {
        maintainAspectRatio: false,
        responsive: true,
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Email Index'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Email Length'
                }
            }
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
                            <RingLoader color="#3B82F6" loading={loading} size={80} />
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
                    <div className="chart-container mb-8" style={{ height: '300px' }}>
                        <Bar data={barData} options={barOptions} />
                    </div>
                    <div className="emails">
                        {currentEmails.map((email, index) => (
                            <div key={index} className={`email ${email.is_phishing ? 'flagged' : ''}`}>
                                <h3>{email.subject}</h3>
                                <p dangerouslySetInnerHTML={{ __html: highlightKeywords(email.body, phishingKeywords) }}></p>
                                <p><strong>Status:</strong> {email.is_phishing ? 'Phishing' : 'Not Phishing'}</p>
                                <p><strong>Score:</strong> {email.score || 'N/A'}</p>  {/* Display score */}
                            </div>
                        ))}
                    </div>
                    <div className="pagination">
                        {Array.from({ length: Math.ceil(emails.length / emailsPerPage) }, (_, i) => (
                            <button
                                key={i}
                                onClick={() => paginate(i + 1)}
                                className={`page-link ${currentPage === i + 1 ? 'active' : ''}`}
                            >
                                {i + 1}
                            </button>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
};

export default Dashboard;
