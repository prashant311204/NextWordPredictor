
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pickle
import json
import os

# Paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
PREDICTORS_FILE = os.path.join(DATA_DIR, 'predictors.npy')
LABEL_FILE = os.path.join(DATA_DIR, 'label.npy')
META_FILE = os.path.join(MODEL_DIR, 'meta.pkl')
MODEL_FILE = os.path.join(MODEL_DIR, 'next_word_model.keras')
HISTORY_FILE = os.path.join(MODEL_DIR, 'history.json')

def train_model():
    print("Loading data...")
    predictors = np.load(PREDICTORS_FILE)
    label = np.load(LABEL_FILE)
    
    with open(META_FILE, 'rb') as f:
        meta = pickle.load(f)
        max_sequence_len = meta['max_sequence_len']
        total_words = meta['total_words']
        
    print(f"Max Sequence Length: {max_sequence_len}")
    print(f"Total Words: {total_words}")
    print(f"Input Shape: {predictors.shape}")
    print(f"Label Shape: {label.shape}")
    
    # Model Architecture
    print("Building model (Lightweight for CPU)...")
    model = Sequential([
        Embedding(total_words, 64, input_length=max_sequence_len-1),
        # Bidirectional is slow on CPU, using standard GRU which is faster than LSTM
        tf.keras.layers.GRU(100), 
        Dropout(0.2),
        Dense(total_words//2, activation='relu'),
        Dense(total_words, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_FILE, monitor='loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.0001)
    
    # Train
    print("Starting training...")
    # Training for fewer epochs but faster
    epochs = 20
    # Increased batch size for speed
    history = model.fit(predictors, label, epochs=epochs, batch_size=128, verbose=1, callbacks=[checkpoint, reduce_lr])
    
    # Save training history
    print("Saving history...")
    history_dict = history.history
    # Convert float32 to float for JSON serialization
    for key in history_dict:
        history_dict[key] = [float(x) for x in history_dict[key]]
        
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_dict, f)
        
    print("Training complete.")

if __name__ == "__main__":
    train_model()
