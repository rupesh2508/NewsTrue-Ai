import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading NewsTruth AI dataset...")

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print("Fake news:", fake.shape)
print("Real news:", true.shape)

fake["label"] = 0
true["label"] = 1

data = pd.concat(
    [fake, true],
    ignore_index=True
)

print("\nTotal news:", len(data))

data["content"] = (
    data["title"].fillna("")
    + " "
    + data["text"].fillna("")
)

X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining news:", len(X_train))
print("Testing news:", len(X_test))

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=50000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed!")

print("Training features:", X_train_tfidf.shape)
print("Testing features:", X_test_tfidf.shape)

print("\nTraining NewsTruth AI model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training completed!")

y_pred = model.predict(
    X_test_tfidf
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========== NEWS TRUTH AI RESULT ==========")

print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake News",
            "Real News"
        ]
    )
)

print("\nSaving NewsTruth AI model...")

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\n========================================")
print("NEWS TRUTH AI MODEL READY")
print("========================================")

print("model.pkl created")
print("vectorizer.pkl created")

print("\nYou can now run the NewsTruth AI Streamlit app.")