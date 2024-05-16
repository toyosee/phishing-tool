Email Phishing Detection Dashboard
Use Case

The Email Phishing Detection Dashboard is a web application designed to analyze emails and identify potential phishing attempts. It integrates with email systems to fetch emails, preprocess their content, and classify them as either phishing or non-phishing using machine learning algorithms. The dashboard provides insights into the total number of emails, flagged emails, and their status.

The application uses a supervised learning algorithm, specifically a Random Forest Classifier, to train a model on the labeled email data. During training, the model learns patterns and relationships in the input data that correlate with the provided output labels.

Potentials

    Enhances email security by automatically detecting and flagging phishing emails.
    Reduces the risk of falling victim to phishing attacks.
    Provides a centralized platform for monitoring and managing suspicious emails.
    Offers valuable insights into email patterns and trends.

Learning Models Used

    Random Forest Classifier: Utilized for email classification into phishing and non-phishing categories.

AI Capability

The application leverages machine learning techniques to analyze email content and predict whether an email is likely to be a phishing attempt. It employs natural language processing (NLP) to preprocess email text and extract relevant features for classification.

Libraries Used

    Python: Used for backend development.
        Flask: Web framework for building the API.
        imaplib: Library for accessing IMAP mailboxes.
        scikit-learn: Library for machine learning tasks.
        pickle: Library for serializing/deserializing Python objects.
        dotenv: Library for loading environment variables from .env file.
    JavaScript (React): Used for frontend development.
        axios: Library for making HTTP requests.
        react: JavaScript library for building user interfaces.
        tailwindcss: Utility-first CSS framework for styling.

How to Run and Deploy

    1. Clone Repository: Clone the repository to your local machine.
       git clone <repository_url>

    2. Install Dependencies: Navigate to the project directory and install the required dependencies for both backend and frontend.
    cd phishing-detection-dashboard - Change directory to front end folder
    npm install  # Install frontend dependencies
    pip install -r requirements.txt  # Install backend dependencies

    3. Set Environment Variables: Create a .env file in the backend directory and set the required environment variables (EMAIL_ID, EMAIL_PASSWORD, EMAIL_FETCH_LIMIT). For large emails, you can fetch all or set numbers of emails to fetch.
    EMAIL_ID=your_email@example.com
    EMAIL_PASSWORD=your_email_password
    EMAIL_FETCH_LIMIT = number

    4. Run Backend: Start the Flask server for the backend.
    python main.py

    5. Run Frontend: In a separate terminal, start the React development server for the frontend.
    npm start

    6. Access Dashboard: Open your browser and navigate to http://localhost:3000 to access the Phishing Detection Dashboard.

    7. Access API response: Open your browser and navigate to http://localhost:5000/fetch_emails to access the Phishing Detection API response.


Deployment

For deployment to production, you can use platforms like Heroku, AWS, or DigitalOcean. Follow the platform-specific instructions for deploying Flask and React applications. Ensure to set environment variables for production deployment and configure necessary security measures.
