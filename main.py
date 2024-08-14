# Author: Elijah Abolaji
# Date: 5/16/2024
# Mail: tyabolaji@gmail.com

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
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

class EmailPhishingDetector:
    def __init__(self, server, email_id, password):
        self.server = server
        self.email_id = email_id
        self.password = password
        self.vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def fetch_emails(self, limit=100):
        """Fetches a limited number of emails from the inbox."""
        try:
            mail = imaplib.IMAP4_SSL(self.server)
            mail.login(self.email_id, self.password)
            mail.select('inbox')
            result, data = mail.search(None, 'ALL')
            inbox_email_ids = data[0].split()[-limit:]

            emails = []
            for e_id in inbox_email_ids:
                result, msg_data = mail.fetch(e_id, '(RFC822)')
                if msg_data and msg_data[0]:
                    msg = email.message_from_bytes(msg_data[0][1])
                    emails.append(msg)

            mail.logout()
            return emails
        except imaplib.IMAP4.error as e:
            print(f'IMAP error: {e}')
            return []
        except Exception as e:
            print(f'Error fetching emails: {e}')
            return []

    def preprocess_email(self, email_message):
        """Preprocess the email message by extracting and cleaning the text content."""
        try:
            subject = email_message['subject'] or ''
            body = ''
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == 'text/plain' and part.get('Content-Disposition') is None:
                        part_payload = part.get_payload(decode=True)
                        if part_payload:
                            body += part_payload.decode('utf-8', errors='ignore')
                    elif part.get_content_type() == 'text/html' and part.get('Content-Disposition') is None:
                        part_payload = part.get_payload(decode=True)
                        if part_payload:
                            soup = BeautifulSoup(part_payload, 'html.parser')
                            body += soup.get_text(separator=' ')
            else:
                body_payload = email_message.get_payload(decode=True)
                if body_payload:
                    if email_message.get_content_type() == 'text/html':
                        soup = BeautifulSoup(body_payload, 'html.parser')
                        body = soup.get_text(separator=' ')
                    else:
                        body = body_payload.decode('utf-8', errors='ignore')

            words = word_tokenize(body.lower())
            filtered_words = [self.stemmer.stem(word) for word in words if word.isalnum() and word not in self.stop_words]

            return subject, ' '.join(filtered_words)
        except Exception as e:
            print(f'Error preprocessing email: {e}')
            return '', ''

    def extract_features(self, emails):
        """Extract features from the preprocessed email bodies using TF-IDF."""
        corpus = [self.preprocess_email(email)[1] for email in emails]
        X = self.vectorizer.fit_transform(corpus)
        return X

    def train_model(self, emails, labels):
        """Train a RandomForest model on the extracted features and labels."""
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
        """Load the saved model and vectorizer from disk."""
        with open('phishing_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)
        return model

    def detect_phishing(self, email_message, model):
        """Detect whether an email is phishing using the trained model."""
        processed_email = self.preprocess_email(email_message)[1]
        email_features = self.vectorizer.transform([processed_email])
        prediction = model.predict(email_features)
        return prediction[0]

@app.route('/fetch_emails', methods=['GET'])
def fetch_emails():
    """Fetch, preprocess, and classify emails, then return the results."""
    server = os.getenv('SERVER_DETAILS')
    email_id = os.getenv('EMAIL_ID')
    password = os.getenv('EMAIL_PASSWORD')
    limit = int(os.getenv('EMAIL_FETCH_LIMIT', 50))

    detector = EmailPhishingDetector(server, email_id, password)
    emails = detector.fetch_emails(limit=limit)

    # Load phishing keywords and scores from an Excel file
    try:
        keywords_df = pd.read_excel('phishingKeywords.xlsx')
        phishing_keywords = keywords_df[['Keywords', 'Score']].dropna().to_dict(orient='records')
    except Exception as e:
        return jsonify({'error': f'Failed to load phishing keywords: {str(e)}'})

    # Calculate phishing scores and labels for each email
    labels = []
    for email_msg in emails:
        processed_email_body = detector.preprocess_email(email_msg)[1]
        # Calculate phishing score based on keywords
        score = sum(keyword['Score'] for keyword in phishing_keywords if keyword['Keywords'].lower() in processed_email_body.lower())
        labels.append(1 if score >= 7 else 0)  # Flag email if score >= 5

    if emails:
        try:
            model = detector.train_model(emails, labels)
        except ValueError as e:
            return jsonify({'error': str(e)})

        flagged_emails = []
        all_emails = []
        email_lengths = []

        for email in emails:
            prediction = detector.detect_phishing(email, model)
            subject, body = detector.preprocess_email(email)
            email_data = {
                'subject': subject,
                'body': body,
                'score': sum(keyword['Score'] for keyword in phishing_keywords if keyword['Keywords'].lower() in body.lower()),  # Include the score
                'is_phishing': bool(prediction)
            }

            email_lengths.append(len(body))
            all_emails.append(email_data)
            if prediction == 1:
                flagged_emails.append(email_data)

        # Only send keywords to the frontend
        keywords_only = [keyword['Keywords'] for keyword in phishing_keywords]

        return jsonify({
            'message': 'Emails fetched successfully',
            'total_emails': len(emails),
            'all_emails': all_emails,
            'flagged_count': len(flagged_emails),
            'flagged_emails': flagged_emails,
            'email_lengths': email_lengths,
            'phishing_keywords': keywords_only
        })

    return jsonify({'message': 'No emails found'})

if __name__ == '__main__':
    app.run(debug=True)
