import streamlit as st
import json
import os
import re
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle as cpickle

# Global variables for models and data
classifier = None
cvv = None
filename = None

# Define Streamlit interface
st.title("Spammer Detection and Fake User Identification on Social Networks")
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose an action:", [
    "Upload Dataset", 
    "Load Naive Bayes", 
    "Detect Fake Content", 
    "Run Algorithms", 
    "Accuracy Comparison"
])

# Helper function to process text
def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

# Step 1: Upload dataset
if option == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload Twitter JSON Dataset", type=["json"])
    if uploaded_file:
        filename = uploaded_file.name
        st.success(f"{filename} uploaded successfully!")
        tweets_data = json.load(uploaded_file)
        st.write("First few records from the dataset:")
        st.json(tweets_data[:5])  # Display a sample

# Step 2: Load Naive Bayes
elif option == "Load Naive Bayes":
    try:
        classifier = cpickle.load(open('model/naiveBayes.pkl', 'rb'))
        cv = CountVectorizer(decode_error="replace", vocabulary=cpickle.load(open("model/feature.pkl", "rb")))
        cvv = CountVectorizer(vocabulary=cv.get_feature_names(), stop_words="english", lowercase=True)
        st.success("Naive Bayes Classifier loaded successfully!")
    except Exception as e:
        st.error(f"Error loading Naive Bayes: {e}")

# Step 3: Detect Fake Content
elif option == "Detect Fake Content":
    if filename and classifier:
        total, fake_acc, spam_acc = 0, 0, 0
        st.info("Analyzing tweets...")
        dataset = "Favourites,Retweets,Following,Followers,Reputation,Hashtag,Fake,Class\n"
        # Iterate through files
        for root, dirs, files in os.walk(filename):
            for file in files:
                with open(os.path.join(root, file), "r") as fdata:
                    data = json.load(fdata)
                    # Extract features
                    tweet_text = re.sub(r'\W+', ' ', data['text'])
                    test = cvv.fit_transform([tweet_text])
                    prediction = classifier.predict(test)
                    # Output results
                    st.write(f"Tweet: {tweet_text}")
                    st.write(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
    else:
        st.warning("Upload a dataset and load Naive Bayes first!")

# Step 4: Run Machine Learning Algorithms
elif option == "Run Algorithms":
    try:
        train = pd.read_csv("features.txt")
        X = train.iloc[:, :-1]
        Y = train.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

        # Random Forest
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred) * 100
        st.success(f"Random Forest Accuracy: {rf_accuracy:.2f}%")

        # Naive Bayes
        nb_model = BernoulliNB(binarize=0.0)
        nb_model.fit(X_train, y_train)
        nb_pred = nb_model.predict(X_test)
        nb_accuracy = accuracy_score(y_test, nb_pred) * 100
        st.success(f"Naive Bayes Accuracy: {nb_accuracy:.2f}%")
    except Exception as e:
        st.error(f"Error running algorithms: {e}")

# Step 5: Accuracy Comparison
elif option == "Accuracy Comparison":
    try:
        accuracies = {"Random Forest": rf_accuracy, "Naive Bayes": nb_accuracy}
        st.bar_chart(accuracies)
    except Exception as e:
        st.error(f"Error generating comparison: {e}")

