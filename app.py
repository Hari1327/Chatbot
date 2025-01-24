import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
@st.cache_resource
def load_chatbot_model():
    model = load_model("chat_model.h5")
    return model

# Load tokenizer and label encoder
@st.cache_resource
def load_support_files():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as ecn_file:
        lbl_encoder = pickle.load(ecn_file)
    return tokenizer, lbl_encoder

# Load model and support files
model = load_chatbot_model()
tokenizer, lbl_encoder = load_support_files()

# Preprocess input text
def preprocess_input(text):
    max_len = 20
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, truncating='post', maxlen=max_len)
    return padded

# Generate chatbot response
def get_response(user_input):
    padded_input = preprocess_input(user_input)
    prediction = model.predict(padded_input, verbose=0)
    predicted_label = np.argmax(prediction)
    tag = lbl_encoder.inverse_transform([predicted_label])[0]

    # Retrieve response from intents file
    with open('Intent.json') as file:
        intents = json.load(file)

    for intent in intents['intents']:
        if intent['intent'] == tag:
            return np.random.choice(intent['responses'])

    return "I don't understand, please try again."

# Streamlit app layout
st.title("AI Chatbot")
st.write("Ask me anything!")

# User input
user_input = st.text_input("You:", placeholder="Type your message here...")

if st.button("Send"):
    if user_input.strip() != "":
        response = get_response(user_input)
        st.write(f"Bot: {response}")
    else:
        st.write("Bot: Please type something!")

