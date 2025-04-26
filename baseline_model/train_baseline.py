import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
data_path = "../data/sofmattress_train.csv"  # Adjust path if needed
df = pd.read_csv(data_path)

# 2. Basic Preprocessing
df['sentence'] = df['sentence'].str.lower().str.strip()

# Label encoding
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['label'])

# Save the class names
intent_classes = label_encoder.classes_

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['sentence'], df['label_id'], test_size=0.2, random_state=42, stratify=df['label_id']
)

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)

# 6. Predict and Evaluate
y_pred = lr_model.predict(X_test_vec)

# Print Metrics
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=intent_classes))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=intent_classes, yticklabels=intent_classes, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
os.makedirs("../results", exist_ok=True)
plt.savefig("../results/baseline_confusion_matrix.png")
plt.show()

# 7. Save Model and Vectorizer (Optional)
import joblib

os.makedirs("../results/baseline_model", exist_ok=True)
joblib.dump(lr_model, "../results/baseline_model/logistic_regression_model.pkl")
joblib.dump(vectorizer, "../results/baseline_model/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "../results/baseline_model/label_encoder.pkl")

print("\nModel, vectorizer, and label encoder saved successfully!")
