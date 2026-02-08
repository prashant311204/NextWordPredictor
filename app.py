
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import plotly.express as px
import os

# Set page config
st.set_page_config(page_title="Next Word Predictor", layout="wide")

# Paths
MODEL_PATH = 'models/next_word_model.keras'
TOKENIZER_PATH = 'models/tokenizer.pkl'
META_PATH = 'models/meta.pkl'
HISTORY_PATH = 'models/history.json'

@st.cache_resource
def load_resources():
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(META_PATH, 'rb') as f:
            meta = pickle.load(f)
        return model, tokenizer, meta
    except Exception as e:
        return None, None, None

def predict_next_words(model, tokenizer, text, max_sequence_len, top_k=5):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    predicted_probs = model.predict(token_list, verbose=0)[0]
    predicted_indices = np.argsort(predicted_probs)[-top_k:][::-1]
    
    results = []
    for idx in predicted_indices:
        word = tokenizer.index_word.get(idx, '<unk>')
        prob = predicted_probs[idx]
        results.append({'word': word, 'probability': prob})
        
    return results

def generate_text(model, tokenizer, seed_text, max_sequence_len, next_words=10):
    text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = token_list[-(max_sequence_len-1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        text += " " + output_word
    return text

# UI Layout
st.title("📚 Next Word Predictor & Generator")
st.markdown("### Deep Learning with LSTM & TensorFlow")

model, tokenizer, meta = load_resources()

if model is None:
    st.error("Model or Tokenizer not found! Please train the model first.")
else:
    max_sequence_len = meta['max_sequence_len']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Predict Next Word")
        input_text = st.text_input("Enter a phrase (e.g., 'The game is'):", "The game is")
        
        if input_text:
            predictions = predict_next_words(model, tokenizer, input_text, max_sequence_len)
            
            st.write("Top Predictions:")
            df = pd.DataFrame(predictions)
            st.dataframe(df)
            
            fig = px.bar(df, x='word', y='probability', title="Prediction Probabilities", color='probability')
            st.plotly_chart(fig)
            
    with col2:
        st.subheader("Generate Text")
        seed_text = st.text_input("Seed Text for Generation:", "Elementary my dear")
        num_words = st.slider("Number of words to generate:", 5, 50, 10)
        
        if st.button("Generate"):
            with st.spinner("Generating..."):
                generated = generate_text(model, tokenizer, seed_text, max_sequence_len, num_words)
                st.success(generated)
                
    st.markdown("---")
    st.subheader("Model Performance")
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            import json
            history = json.load(f)
            
        acc = history.get('accuracy', [])
        loss = history.get('loss', [])
        epochs = list(range(1, len(acc) + 1))
        
        metrics_df = pd.DataFrame({'Epoch': epochs, 'Accuracy': acc, 'Loss': loss})
        
        st.line_chart(metrics_df.set_index('Epoch'))
    else:
        st.info("Training history not available yet.")

