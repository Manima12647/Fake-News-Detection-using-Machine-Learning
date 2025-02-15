import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

# Download stopwords if not available
nltk.download('stopwords')

# Initialize Porter Stemmer
port_stem = PorterStemmer()

# Function for text preprocessing (stemming + stopword removal)
def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)  # Remove non-alphabet characters
    con = con.lower().split()  # Convert to lowercase and split
    con = [port_stem.stem(word) for word in con if word not in stopwords.words('english')]  # Stemming
    return ' '.join(con)  # Join words back into a sentence

# Train and save the model if files don't exist
try:
    vector_form = pickle.load(open("vector.pkl", "rb"))
    load_model = pickle.load(open("model.pkl", "rb"))
    print("Model and Vectorizer loaded successfully.")
except (FileNotFoundError, EOFError):
    print("Training new model...")

    # Sample dataset (Replace with actual dataset)
    data = ["Fake news example", "Another reliable news", "Misleading information spread"]
    labels = [1, 0, 1]  # 1 = Fake, 0 = Real

    # Train TF-IDF Vectorizer
    vector_form = TfidfVectorizer()
    X = vector_form.fit_transform(data)

    # Train Decision Tree Model
    load_model = DecisionTreeClassifier()
    load_model.fit(X, labels)

    # Save trained vectorizer and model
    pickle.dump(vector_form, open("vector.pkl", "wb"))
    pickle.dump(load_model, open("model.pkl", "wb"))
    print("New model and vectorizer trained and saved.")

# Fake news prediction function
def fake_news(news):
    news = stemming(news)  # Preprocess input text
    input_data = [news]
    vectorized_input = vector_form.transform(input_data)  # Convert to TF-IDF
    prediction = load_model.predict(vectorized_input)  # Predict using model
    return prediction[0]

# Streamlit App UI
st.title('📰 Fake News Classification App')
st.subheader("Input the News Content Below")

sentence = st.text_area("Enter your news content here:", "", height=200)
predict_btt = st.button("Predict")

if predict_btt:
    if sentence.strip():  # Ensure input is not empty
        prediction_class = fake_news(sentence)
        if prediction_class == 0:
            st.success('✅ Reliable News')
        else:
            st.warning('⚠️ Unreliable (Fake) News')
    else:
        st.error("Please enter some text to classify.")
