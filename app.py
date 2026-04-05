import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. THE EXACT ARCHITECTURE (From your error log)
def build_model_skeleton():
    model = Sequential([
        # input_dim=28448, output_dim=128, input_length=43
        Embedding(input_dim=28448, output_dim=128, input_length=43),
        # units=256
        LSTM(256),
        # units=28448
        Dense(28448, activation='softmax')
    ])
    return model

@st.cache_resource
def load_my_model():
    try:
        model = build_model_skeleton()
        # We use the .keras file but only extract the weights
        # This skips the 'quantization_config' error entirely!
        model.load_weights('lstm_model.keras')
        return model
    except Exception as e:
        st.error(f"Manual load failed: {e}")
        return None

model = load_my_model()

# 2. LOAD TOKENIZER
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

reverse_index = {idx: word for word, idx in tokenizer.word_index.items()}

# 3. MATCH THE LENGTH (From your log: build_input_shape: [None, 43])
MAX_SEQ_LEN = 43 

def generate_text(seed_text, num_words=10):
    if model is None:
        return "Error: Model not loaded."
    
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

# --- Streamlit UI ---
st.title("Next Word Prediction")
seed_text = st.text_input("Enter a starting text:")
num_words = st.slider("Number of words to generate:", 1, 20, 10)

if st.button("Generate"):
    if seed_text:
        with st.spinner('Thinking...'):
            result = generate_text(seed_text, num_words)
            st.subheader("Generated Text:")
            st.write(result)
