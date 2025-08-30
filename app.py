import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .spam-prediction {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin: 20px 0;
    }
    .ham-prediction {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 20px 0;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">📧 Email Spam Detection</h1>', unsafe_allow_html=True)

# Load the model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open('spam_classifier_complete.pkl', 'rb') as file:
            model_info = pickle.load(file)
        return model_info['model'], model_info['vectorizer']
    except:
        st.error("Model file not found. Please make sure 'spam_classifier_complete.pkl' is in the same directory.")
        return None, None

model, vectorizer = load_model()

# Function to clean text
def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(' +', ' ', text)
    text = text.strip()
    return text

# Function to predict spam
def predict_spam(email_text):
    if model is None or vectorizer is None:
        return "Error: Model not loaded"
    
    # Clean the text
    cleaned_text = clean_text(email_text)
    
    # Transform the text
    features = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)
    
    # Return result
    return "SPAM" if prediction[0] == 0 else "HAM", prediction_proba

# Sample spam and ham emails
sample_spam = ["Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize now!",
               "Your bank account has been compromised. Please verify your details immediately to secure your account."]

sample_ham = ["Hey, are we still meeting for lunch tomorrow?",
              "Hi, just checking in to see how you're doing. Let's catch up soon."]

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This app uses a machine learning model to detect spam emails. 
    The model was trained on a dataset of spam and ham (non-spam) emails.
    """)
    
    st.header("Try Sample Emails")
    if st.button("Load Spam Example"):
        st.session_state.email_text = sample_spam[0]
    if st.button("Load Ham Example"):
        st.session_state.email_text = sample_ham[0]
    
    st.header("Model Information")
    if model is not None:
        st.success("Model loaded successfully!")
        st.write("Algorithm: Logistic Regression")
    else:
        st.error("Model not loaded")

# Initialize session state for email text
if 'email_text' not in st.session_state:
    st.session_state.email_text = ""

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Email Text")
    email_input = st.text_area(
        "Paste the email content below:",
        height=200,
        value=st.session_state.email_text,
        key="email_input"
    )
    
    predict_button = st.button("Check for Spam", type="primary")
    
    if predict_button and email_input.strip():
        with st.spinner("Analyzing..."):
            result, probabilities = predict_spam(email_input)
            
            if "Error" not in result:
                spam_prob = probabilities[0][0] * 100
                ham_prob = probabilities[0][1] * 100
                
                st.subheader("Prediction Result")
                
                if result == "SPAM":
                    st.markdown(f"""
                    <div class="spam-prediction">
                        <h2>🚫 This email is likely SPAM</h2>
                        <p>Confidence: {spam_prob:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ham-prediction">
                        <h2>✅ This email is likely HAM (not spam)</h2>
                        <p>Confidence: {ham_prob:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display probability chart
                prob_df = pd.DataFrame({
                    'Category': ['SPAM', 'HAM'],
                    'Probability': [spam_prob, ham_prob]
                })
                
                st.bar_chart(prob_df.set_index('Category'), use_container_width=True)

with col2:
    st.subheader("How It Works")
    st.write("""
    1. Paste the email text in the input box
    2. Click the 'Check for Spam' button
    3. The model will analyze the text
    4. Results will show if it's spam or not
    """)
    
    st.subheader("Tips to Identify Spam")
    st.info("""
    - Urgent requests for personal information
    - Too-good-to-be-true offers
    - Poor grammar and spelling
    - Unknown senders asking for money
    - Suspicious links or attachments
    """)
    
    st.subheader("Accuracy Note")
    st.warning("""
    No spam detection system is 100% accurate. 
    Always use caution with suspicious emails, 
    even if they're marked as safe.
    """)

# Footer
st.markdown("---")
st.markdown('<p class="footer">Email Spam Detection App © 2023 | Built with Streamlit and Scikit-learn</p>', unsafe_allow_html=True)