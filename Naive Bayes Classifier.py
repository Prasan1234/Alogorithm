import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
dataset = [
    ("I love this movie", "positive"),
    ("This movie is great", "positive"),
    ("I dislike this movie", "negative"),
    ("This movie is terrible", "negative"),
    ("I don't like this movie", "negative")
]

# Split dataset into features (X) and labels (y)
X, y = zip(*dataset)

# Create a vectorizer to convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset into training and testing sets
split_ratio = 0.8  # 80% training, 20% testing
split_index = int(len(X) * split_ratio)
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
