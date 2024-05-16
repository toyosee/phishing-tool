import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { RingLoader } from 'react-spinners'; // Import loading animation
import './Dashboard.css';
// import './index.css'; // Import Tailwind CSS styles


const Dashboard = () => {
    const [emails, setEmails] = useState([]);
    const [totalEmails, setTotalEmails] = useState(0);
    const [flaggedCount, setFlaggedCount] = useState(0);
    const [loading, setLoading] = useState(true); // State for loading animation

    useEffect(() => {
        axios.get('http://localhost:5000/fetch_emails')
            .then(response => {
                console.log(response.data); // Log the response for debugging
                setEmails(response.data.all_emails);
                setTotalEmails(response.data.total_emails);
                setFlaggedCount(response.data.flagged_count);
                setLoading(false); // Turn off loading animation
            })
            .catch(error => {
                console.error('Error fetching emails:', error);
                setLoading(false); // Turn off loading animation even in case of error
            });
    }, []);

    return (
        <div className="dashboard">
            <h1 className="text-4xl mb-4">Phishing Detection Dashboard</h1>
            <div className="stats mb-8">
                <div className="stat-item">
                    <div className="stat-title">Total Emails</div>
                    <div className="stat-value">{totalEmails}</div>
                </div>
                <div className="stat-item">
                    <div className="stat-title">Flagged Emails</div>
                    <div className="stat-value">{flaggedCount}</div>
                </div>
            </div>
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
                <div className="emails">
                    {emails.map((email, index) => (
                        <div key={index} className={`email ${email.is_phishing ? 'flagged' : ''}`}>
                            <h3>{email.subject}</h3>
                            <p>{email.body}</p>
                            <p>Status: {email.is_phishing ? 'Phishing' : 'Not Phishing'}</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default Dashboard;
