"""
Author: Elijah Abolaji
Date: 5/16/2024
Mail: tyabolaji@gmail.com
"""

# This script Predicts phishing emails using AI/ML

import imaplib
import email
from email import policy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class EmailPhishingDetector:
    def __init__(self, server, email_id, password):
        self.server = server
        self.email_id = email_id
        self.password = password
        self.vectorizer = TfidfVectorizer()

    def fetch_emails(self, limit=100):
        try:
            mail = imaplib.IMAP4_SSL(self.server)
            mail.login(self.email_id, self.password)
            mail.select('inbox')

            result, data = mail.search(None, 'ALL')
            if data is None or data[0] is None:
                print('No emails found.')
                return []

            email_ids = data[0].split()
            email_ids = email_ids[-limit:]  # Fetch only the last 'limit' number of emails
            emails = []
            for e_id in email_ids:
                result, msg_data = mail.fetch(e_id, '(RFC822)')
                if msg_data is None or msg_data[0] is None:
                    print(f'No data for email id {e_id}')
                    continue

                msg = email.message_from_bytes(msg_data[0][1])
                emails.append(msg)

            mail.logout()
            return emails
        except Exception as e:
            print(f'Error fetching emails: {e}')
            return []

    def preprocess_email(self, email_message):
        try:
            subject = email_message['subject'] if email_message['subject'] else ''
            body = ''
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == 'text/plain' and part.get('Content-Disposition') is None:
                        part_payload = part.get_payload(decode=True)
                        if part_payload:
                            body += part_payload.decode('utf-8', errors='ignore')
            else:
                body_payload = email_message.get_payload(decode=True)
                if body_payload:
                    body = body_payload.decode('utf-8', errors='ignore')
            return subject, body
        except Exception as e:
            print(f'Error preprocessing email: {e}')
            return '', ''

    def extract_features(self, emails):
        corpus = [self.preprocess_email(email)[1] for email in emails]  # Use body for feature extraction
        X = self.vectorizer.fit_transform(corpus)
        return X

    def train_model(self, emails, labels):
        if len(emails) != len(labels):
            raise ValueError("Number of emails and labels must be the same")

        X = self.extract_features(emails)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

        # Save the model and vectorizer
        with open('phishing_model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        with open('vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)

        return model

    def load_model(self):
        with open('phishing_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)
        return model

    def detect_phishing(self, email_message, model):
        processed_email = self.preprocess_email(email_message)[1]  # Use body for prediction
        email_features = self.vectorizer.transform([processed_email])
        prediction = model.predict(email_features)
        return prediction[0]

@app.route('/fetch_emails', methods=['GET'])
def fetch_emails():
    server = os.getenv('SERVER_DETAILS')
    email_id = os.getenv('EMAIL_ID')  # Getting email from environment variables
    password = os.getenv('EMAIL_PASSWORD')
    limit = int(os.getenv('EMAIL_FETCH_LIMIT', 50))  # Get the limit from .env file, default to 50

    detector = EmailPhishingDetector(server, email_id, password)
    emails = detector.fetch_emails(limit=limit)
    
    # Generate labels dynamically (Example: all non-phishing or a mix for testing)
    labels = [0] * len(emails)  # Example: All emails are non-phishing
    # or
    # labels = [0 if i % 2 == 0 else 1 for i in range(len(emails))]  # Example: Alternate between non-phishing and phishing

    if emails:
        try:
            model = detector.train_model(emails, labels)
        except ValueError as e:
            return jsonify({'error': str(e)})

        flagged_emails = []
        all_emails = []
        for email in emails:
            prediction = detector.detect_phishing(email, model)
            subject, body = detector.preprocess_email(email)
            email_data = {
                'subject': subject,
                'body': body,
                'is_phishing': bool(prediction)
            }
            all_emails.append(email_data)
            if prediction == 1:
                flagged_emails.append(email_data)

        return jsonify({
            'message': 'Emails fetched successfully',
            'total_emails': len(emails),
            'all_emails': all_emails,
            'flagged_count': len(flagged_emails),
            'flagged_emails': flagged_emails
        })

    return jsonify({'message': 'No emails found'})

if __name__ == '__main__':
    app.run(debug=True)
