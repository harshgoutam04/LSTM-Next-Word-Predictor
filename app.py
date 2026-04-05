import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load the model the easy way
@st.cache_resource
def load_my_model():
    try:
        # Notice the change in extension to .keras
        return load_model('lstm_model.keras')
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        return None

model = load_my_model()

# 2. Load Tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

reverse_index = {idx: word for word, idx in tokenizer.word_index.items()}

# Match this to your training (usually 1 less than the 'input_shape' seen in logs)
MAX_SEQ_LEN = 43 

def generate_text(seed_text, num_words=10):
    if model is None:
        return "Error: Model is not loaded."
    
    text = seed_text
    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]
        padded = pad_sequences([seq], maxlen=MAX_SEQ_LEN, padding='pre')
        
        preds = model.predict(padded, verbose=0)[0]
        pos = np.argmax(preds)
        next_word = reverse_index.get(pos, '')
        
        if not next_word:
            break
        text += ' ' + next_word
    return text

# UI
st.title("Next Word Prediction")
seed_text = st.text_input("Enter a starting text:")
num_words = st.slider("Number of words to generate:", 1, 20, 10)

if st.button("Generate"):
    if seed_text:
        with st.spinner('Predicting...'):
            result = generate_text(seed_text, num_words)
            st.subheader("Generated Text:")
            st.write(result)
