import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 🔹 Load dataset
df = pd.read_csv("spam.csv", encoding='latin1')  # use 'utf-8' if you get decode errors

# 🔹 Clean & standardize columns
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df[['label', 'message']]
df = df[df['label'].isin(['ham', 'spam'])]

# 🔹 Map labels to binary (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 🔹 Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 🔹 Create model pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# 🔹 Train the model
pipeline.fit(X_train, y_train)

# 🔹 Predict & evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("✅ Accuracy on test data:", round(acc * 100, 2), "%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# 🔹 Save the pipeline as a dictionary
os.makedirs("model", exist_ok=True)
joblib.dump({
    "model": pipeline,
    "vectorizer": pipeline.named_steps["vectorizer"]
}, "model/spam_model.pkl")

print("\n📁 Model saved as 'model/spam_model.pkl'")
