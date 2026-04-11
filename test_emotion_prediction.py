import nltk
import ssl
import certifi
from nltk.corpus import stopwords

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
import re
import pandas as pd

# Load the saved LSTM model
loaded_lstm_model = load_model('lstm_emotion_model.h5')
print("LSTM model loaded successfully.")
cv=CountVectorizer(max_features=5000)
import numpy as np
y=[]
from nltk.stem import PorterStemmer
ps=PorterStemmer()
def stem_words(text):
  # This function is not used in the current predict_emotion, but kept for consistency if needed elsewhere.
  for i in text:
    y.append(ps.stem(i))
  z=y[:]
  y.clear()
  return z
def clean_html(text):
  clean=re.compile('<.*?>')
  return re.sub(clean,'',text)
def remove_special(text):
  x=''
  for i in text:
    if i.isalnum():
      x=x+i
    else:
      x=x+' '
  return x
def predict_emotion(text, cv, ps, stopwords_list, lstm_model):
    # Apply cleaning
    text = clean_html(text)
    text = remove_special(text)

    # Tokenize and remove stopwords
    tokens = text.split()
    filtered_tokens = [i for i in tokens if i not in stopwords_list]

    # Stem words
    stemmed_tokens = [ps.stem(i) for i in filtered_tokens]

    # Join back
    processed_text = " ".join(stemmed_tokens)

    # Transform using CountVectorizer
    vectorized_text = cv.transform([processed_text]).toarray()

    # Reshape for LSTM input
    reshaped_text = vectorized_text.reshape(1, 1, vectorized_text.shape[1])

    # Predict using the LSTM model
    prediction = lstm_model.predict(reshaped_text)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return predicted_class

# Example usage:
new_text = "i feel to have these amazing people in my life"
english_stopwords = stopwords.words('english')

# --- FIX: Fit the CountVectorizer with training data ---
# Load the training data
# Assuming 'training.csv' contains a 'text' column that was used to train the model.
# If your training data is stored differently or the text column has another name,
# please adjust this section accordingly.
try:
    df = pd.read_csv('training.csv')
    print(f"Loaded training data of shape: {df.shape}") # Debug print
    # Preprocess training data similarly to prediction data
    # Apply clean_html and remove_special functions first
    df['processed_text'] = df['text'].apply(clean_html).apply(remove_special)

    # Tokenize, remove stopwords, and stem for fitting the CountVectorizer
    processed_corpus = []
    for text in df['processed_text']:
        tokens = text.split()
        filtered_tokens = [i for i in tokens if i not in english_stopwords]
        stemmed_tokens = [ps.stem(i) for i in filtered_tokens]
        processed_corpus.append(" ".join(stemmed_tokens))
    print(f"Length of processed_corpus: {len(processed_corpus)}") # Debug print

    # Fit the CountVectorizer on the processed training corpus
    cv.fit(processed_corpus)
    print("CountVectorizer fitted successfully with training data.")
    # Now vocabulary_ attribute is available
    print(f"CountVectorizer vocabulary size AFTER fit: {len(cv.vocabulary_)}") # Critical debug print
except FileNotFoundError:
    print("Error: 'training.csv' not found. Please ensure your training data is available and correctly named.")
    print("You need to manually fit the CountVectorizer with the data your model was trained on.")
    # Fix: Fit cv with a dummy vocabulary to prevent NotFittedError, but note this will impact prediction quality.
    # Ideally, load a pre-fitted CountVectorizer or the original training data.
    dummy_vocabulary_list = [f"placeholder_word_{i}" for i in range(5000)]
    cv.fit(dummy_vocabulary_list)
    print("CountVectorizer fitted with a dummy vocabulary due to missing training.csv.")
except KeyError:
    print("Error: 'text' column not found in training.csv. Please ensure the column containing text data is named 'text' or update the code accordingly.")
    print("You need to manually fit the CountVectorizer with the data your model was trained on.")

# Now, transform the new text using the correctly fitted CountVectorizer
X=cv.transform([new_text]).toarray()

# You can also define a mapping for labels if you know them
emotion_labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

predicted_label_index = predict_emotion(new_text, cv, ps, english_stopwords, loaded_lstm_model)
predicted_emotion = emotion_labels.get(predicted_label_index, 'Unknown')

print(f"The predicted emotion for the text \"{new_text}\" is: {predicted_emotion}")
print("Predicted Output:",predicted_emotion)