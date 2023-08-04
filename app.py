import streamlit as st
import pickle

# Load the pre-trained model pipeline
with open('email_spam_pipeline.pkl', 'rb') as model_file:
    model_pipeline = pickle.load(model_file)

# Streamlit app header
st.title("Email Spam Detection App")

# Text input for user to enter an email
user_input = st.text_area("Enter an email:", "")

# Button to initiate prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter an email.")
    else:
        # Make prediction using the model pipeline
        prediction = model_pipeline.predict([user_input])
        prediction_prob = model_pipeline.predict_proba([user_input])

        # Display the prediction result
        if prediction[0] == 1:
            st.error("Spam Detected!")
        else:
            st.success("Not Spam")

        st.write("Prediction Probability (Not Spam, Spam):", prediction_prob)

# Streamlit footer
st.write("Note: This is a simple demo for email spam detection.")
