import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('phishing_model.pkl')

# Streamlit title and description
st.title('Phishing URL Detection')
st.write('Enter the details of the URL to detect whether it is phishing or legitimate.')

# Define the input fields for the required features
URLLength = st.number_input('URL Length', min_value=1)
DomainLength = st.number_input('Domain Length', min_value=1)
URLSimilarityIndex = st.number_input('URL Similarity Index (0-1)', min_value=0.0, max_value=1.0, step=0.01)
CharContinuationRate = st.number_input('Character Continuation Rate (0-1)', min_value=0.0, max_value=1.0, step=0.01)
URLCharProb = st.number_input('URL Character Probability', min_value=0.0, max_value=1.0, step=0.01)
NoOfLettersInURL = st.number_input('Number of Letters in URL', min_value=0)
LetterRatioInURL = st.number_input('Letter Ratio in URL', min_value=0.0, max_value=1.0, step=0.01)
DegitRatioInURL = st.number_input('Digit Ratio in URL', min_value=0.0, max_value=1.0, step=0.01)
NoOfOtherSpecialCharsInURL = st.number_input('Number of Special Characters in URL', min_value=0)
SpacialCharRatioInURL = st.number_input('Special Character Ratio in URL', min_value=0.0, max_value=1.0, step=0.01)
IsHTTPS = st.selectbox('Is HTTPS', [0, 1])
LineOfCode = st.number_input('Lines of Code', min_value=1)
HasTitle = st.selectbox('Has Title', [0, 1])
DomainTitleMatchScore = st.number_input('Domain Title Match Score (0-1)', min_value=0.0, max_value=1.0, step=0.01)
URLTitleMatchScore = st.number_input('URL Title Match Score (0-1)', min_value=0.0, max_value=1.0, step=0.01)
HasFavicon = st.selectbox('Has Favicon', [0, 1])
Robots = st.selectbox('Has Robots.txt', [0, 1])
IsResponsive = st.selectbox('Is Responsive', [0, 1])
HasDescription = st.selectbox('Has Description', [0, 1])
NoOfiFrame = st.selectbox('Number of iFrames', [0, 1])
HasSocialNet = st.selectbox('Has Social Network Links', [0, 1])
HasSubmitButton = st.selectbox('Has Submit Button', [0, 1])
NoOfSelfRef = st.number_input('Number of Self-References', min_value=0)
HasHiddenFields = st.selectbox('Has Hidden Fields', [0, 1])
Pay = st.selectbox('Has Payment Gateway', [0, 1])
HasCopyrightInfo = st.selectbox('Has Copyright Info', [0, 1])
NoOfImage = st.number_input('Number of Images', min_value=0)
NoOfJS = st.number_input('Number of JavaScript Files', min_value=0)
NoOfExternalRef = st.number_input('Number of External References', min_value=0)
label = st.selectbox('Label', [0, 1])
# Predict function
if st.button('Predict'):
    try:
        # Prepare features for prediction
        features = np.array([
            URLLength, DomainLength, URLSimilarityIndex, CharContinuationRate, URLCharProb,
            NoOfLettersInURL, LetterRatioInURL, DegitRatioInURL, NoOfOtherSpecialCharsInURL, SpacialCharRatioInURL,
            IsHTTPS, LineOfCode, HasTitle, DomainTitleMatchScore, URLTitleMatchScore, HasFavicon, Robots,
            IsResponsive, HasDescription, NoOfiFrame, HasSocialNet, HasSubmitButton, NoOfSelfRef, HasHiddenFields,
            Pay, HasCopyrightInfo, NoOfImage, NoOfJS, NoOfExternalRef, label
        ]).reshape(1, -1)

        # Predict using the loaded model
        prediction = model.predict(features)

        # Output the prediction
        if prediction == 0:
            st.success("The URL is likely legitimate.")
        else:
            st.error("The URL is likely a phishing attempt.")

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

