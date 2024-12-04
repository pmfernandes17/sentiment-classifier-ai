import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    "text": ["I love this product", "This is terrible", "Not bad", "Amazing experience", "Worst ever"],
    "sentiment": ["positive", "negative", "neutral", "positive", "negative"]
}
df = pd.DataFrame(data)

# Preprocessing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    return model.predict(text_vector)[0]

# Example usage
print(predict_sentiment("I hated it"))  # Output: negative
