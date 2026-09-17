import streamlit as st
import joblib

# Load model
model = joblib.load("emotion_model.pkl")
    


# Load TF-IDF vectorizer
vectorizer = joblib.load(
    "tfidf_vectorizer.pkl"
)

# Load emotion mapping
emotion_numbers = joblib.load(
   "emotion_mapping.pkl"
)

# Reverse the mapping
reverse_mapping = {v: k for k, v in emotion_numbers.items()}


# Title
st.title("😊 Emotion Detection using NLP")

# Input
text = st.text_area("Enter a sentence:")

# Button
if st.button("Predict Emotion"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        # Convert input text into TF-IDF
        text_vector = vectorizer.transform([text])

        # Predict emotion number
        prediction = model.predict(text_vector)[0]

        # Convert number → emotion word
        emotion = reverse_mapping[prediction]

        # Show result
        st.success(f"Predicted Emotion: {emotion}")
