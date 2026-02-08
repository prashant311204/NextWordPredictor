# Next Word Predictor & Generator 📚

A Deep Learning project that predicts the next word in a sequence and generates coherent text using LSTM (Long Short-Term Memory) neural networks. The model is trained on the Sherlock Holmes dataset.

## Features
- **Next Word Prediction**: Input a phrase, and the model predicts the most likely next words with probabilities.
- **Text Generation**: Generate complete sentences or paragraphs starting from a seed phrase.
- **Interactive Dashboard**: Built with Streamlit for a user-friendly experience.
- **Visualizations**: Probability distribution charts and training performance metrics.

## Project Structure
- `data/`: Contains the specific dataset and processed numpy arrays.
- `models/`: Stores the trained `.keras` model, tokenizer, and training history.
- `src/`: Source code for preprocessing and training.
- `app.py`: Streamlit dashboard application.

## Installation

1. **Clone the repository** (or download files).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
To retrain the model from scratch:
```bash
python src/train.py
```
*Note: This may take some time depending on your hardware.*

### 2. Running the Dashboard
To launch the interactive application:
```bash
streamlit run app.py
```

## Model Architecture
- **Embedding Layer**: Converts words into dense vectors.
- **Bidirectional LSTM**: Captures context from both past and future (in sequence processing).
- **Dropout**: Prevents overfitting.
- **Dense Output**: Predicts the probability of the next word from the vocabulary.
