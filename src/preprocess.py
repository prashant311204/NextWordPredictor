import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Parameters
DATA_FILE = 'data/sherlock.txt'
TOKENIZER_FILE = 'models/tokenizer.pkl'
SEQUENCE_DATA_FILE = 'data/sequences.npy'
MAX_SEQUENCE_LEN = 20  # Length of input sequence (predict next word after 19 words)

def load_data(data_dir):
    text = ""
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            print(f"Loading {file_path}...")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text += f.read() + "\n"
    return text

def preprocess_data():
    print("Loading data...")
    text = load_data(DATA_DIR)
    
    # Basic cleaning not strictly needed as Tokenizer handles lowercasing and punctuation
    # But let's lowercase just to be sure before split
    corpus = text.lower().split("\n")
    
    # Tokenization
    print("Tokenizing...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    print(f"Total words: {total_words}")
    
    # Create input sequences
    print("Creating sequences...")
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
            
    # Pad sequences
    print("Padding sequences...")
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    # Create predictors and label
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    # label = tf.keras.utils.to_categorical(label, num_classes=total_words) # REMOVED to save memory

    
    # Save artifacts
    print("Saving artifacts...")
    with open(TOKENIZER_FILE, 'wb') as f:
        pickle.dump(tokenizer, f)
        
    np.save('data/predictors.npy', predictors)
    np.save('data/label.npy', label)
    
    # Save max_sequence_len helps during inference
    with open('models/meta.pkl', 'wb') as f:
        pickle.dump({'max_sequence_len': max_sequence_len, 'total_words': total_words}, f)

    print("Data processing complete.")

if __name__ == "__main__":
    preprocess_data()
