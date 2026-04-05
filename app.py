import streamlit as st
import pickle
import numpy as np
import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Embedding, LSTM, Dense
from tf_keras.preprocessing.sequence import pad_sequences

# 1. Define your architecture MANUALLY
def build_model_skeleton(vocab_size, max_len):
    model = Sequential()
    # Using the same parameters from your previous attempt
    model.add(Embedding(vocab_size, 128, input_length=max_len))
    model.add(LSTM(256))
    model.add(Dense(vocab_size, activation='softmax'))
    return model

@st.cache_resource
def load_my_model():
    try:
        # Match these to your training exactly
        VOCAB_SIZE = 284448 
        SEQ_LENGTH = 43   # This is the 'input_length' the model expects
        
        model = build_model_skeleton(VOCAB_SIZE, SEQ_LENGTH)
        model.load_weights('lstm_model.h5')
        return model
    except Exception as e:
        st.error(f"Manual load failed: {e}")
        return None

model = load_my_model()

# Load Tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

reverse_index = {idx: word for word, idx in tokenizer.word_index.items()}

# 2. MATCH THE LENGTHS
# If your model was built with input_length=43, you MUST pad to 43.
MAX_SEQ_LEN = 43 

def generate_text(seed_text, num_words=10):
    if model is None:
        return "Error: Model is not loaded."
    
    text = seed_text
    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]
        # Make sure maxlen here matches the input_length above
        padded = pad_sequences([seq], maxlen=MAX_SEQ_LEN, padding='pre')
        
        preds = model.predict(padded, verbose=0)[0]
        pos = np.argmax(preds)
        next_word = reverse_index.get(pos, '')
        
        if not next_word: # Stop if we can't find a word
            break
        text += ' ' + next_word
    return text

# UI Setup
st.title("Next Word Prediction")
seed_text = st.text_input("Enter a starting text:")
num_words = st.slider("Number of words to generate:", 1, 20, 10)

if st.button("Generate"):
    if seed_text:
        result = generate_text(seed_text, num_words)
        st.subheader("Generated Text:")
        st.write(result)
    else:
        st.warning("Please enter some text first!")
