import streamlit as st
import pickle
import numpy as np
import tensorflow
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tf_keras as keras

def build_model_skeleton(vocab_size, sequence_length):
    model = Sequential()
    # Replace 100 with your actual embedding dimension if different
    model.add(Embedding(vocab_size, 100, input_length=sequence_length))
    # Replace 150 with your actual LSTM units if different
    model.add(LSTM(150))
    model.add(Dense(vocab_size, activation='softmax'))
    return model

@st.cache_resource
def load_my_model():
    try:
        # Update these numbers to match your training settings!
        # Based on your previous error, your sequence length was 43
        VOCAB_SIZE = 2828 # Check your tokenizer length!
        SEQ_LENGTH = 43   
        
        model = build_model_skeleton(VOCAB_SIZE, SEQ_LENGTH)
        
        # Load ONLY the weights, ignoring the problematic config metadata
        model.load_weights('lstm_model.h5')
        return model
    except Exception as e:
        st.error(f"Manual load failed: {e}")
        return None

model = load_my_model()
    
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

reverse_index = {idx: word for word, idx in tokenizer.word_index.items()}

max_len=44

def generate_text(seed_text, num_words=10):
    text=seed_text
    if model is None:
        return "Error: Model is not loaded. Check the app logs."
    for _ in range(num_words):
        seq=tokenizer.texts_to_sequences([text])[0]
        padded=pad_sequences([seq], maxlen=max_len, padding='pre')
        preds=model.predict(padded, verbose=0)[0]
        pos=np.argmax(preds)
        next_word=reverse_index.get(pos, '')
        text += ' ' + next_word
    return text

st.title("Next Word Prediction")
seed_text = st.text_input("Enter a starting text:")
num_words = st.slider("Number of words to generate:", 1, 20, 10)

if st.button("Generate"):
    result = generate_text(seed_text, num_words)
    st.subheader("Generated Text:")
    st.write(result)
