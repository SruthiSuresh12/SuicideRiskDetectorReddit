import pandas as pd

df = pd.read_csv("reddit_posts_preprocessed.csv")
df = df.drop_duplicates(subset=["post_id"])
df = df.dropna(subset=["title", "post_text"])
df = df[(df["title"].str.strip() != "") | (df["post_text"].str.strip() != "")]
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["neutral", "high-risk"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
import joblib

joblib.dump(clf, "suicide_risk_logreg_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
