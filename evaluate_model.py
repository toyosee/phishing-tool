import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import imaplib
import email
from email import policy
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from dotenv import load_dotenv
import os
import pickle

# Load environment variables from .env file
load_dotenv()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define the EmailPhishingDetector class
class EmailPhishingDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

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
        X = self.vectorizer.transform(corpus)
        return X

    def fetch_emails(self, limit=1000):
        """Fetch a limited number of emails from the inbox."""
        server = os.getenv('SERVER_DETAILS')
        email_id = os.getenv('EMAIL_ID')
        password = os.getenv('EMAIL_PASSWORD')

        try:
            mail = imaplib.IMAP4_SSL(server)
            mail.login(email_id, password)
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

    def load_model(self):
        """Load the saved model and vectorizer from disk."""
        with open('phishing_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)
        return model

# Initialize detector
detector = EmailPhishingDetector()

# Fetch emails
emails = detector.fetch_emails(limit=1000)

# Load the trained model and vectorizer
model = detector.load_model()

# Load phishing keywords and scores from an Excel file
try:
    keywords_df = pd.read_excel('phishingKeywords.xlsx')
    phishing_keywords = keywords_df[['Keywords', 'Score']].dropna().to_dict(orient='records')
except Exception as e:
    raise Exception(f'Failed to load phishing keywords: {str(e)}')

# Calculate phishing scores and labels
labels = []
for email_msg in emails:
    processed_email_body = detector.preprocess_email(email_msg)[1]
    score = sum(keyword['Score'] for keyword in phishing_keywords if keyword['Keywords'].lower() in processed_email_body.lower())
    labels.append(1 if score >= 7 else 0)  # Flag email if score >= 5

# Extract features from emails
X = detector.extract_features(emails)

# Split the data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Evaluate the model
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Save the score results to a text file
with open('score_results.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'F1 Score: {f1}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'ROC AUC: {roc_auc}\n')

# Print to console as well
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'ROC AUC: {roc_auc}')

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('roc_curve.png')
plt.show()

# Plot Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_vals, precision_vals)
plt.figure(figsize=(10, 6))
plt.plot(recall_vals, precision_vals, label=f'Precision-Recall Curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid()
plt.savefig('precision_recall_curve.png')
plt.show()
