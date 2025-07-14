from flask import Flask, request, jsonify, render_template
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import praw

# Initialize Reddit API (use environment variables or config for secrets in production)
reddit = praw.Reddit(
    client_id="2eX3T9gzL07MzGepVqB_-A",
    client_secret="VS-2e-FyNOoXKMC5YttUOe4CJHiw0A",
    user_agent="DepressionWatcher"
)

app = Flask(__name__)

# Load model and vectorizer
clf = joblib.load("suicide_risk_logreg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def extract_post_from_url(url):
    submission = reddit.submission(url=url)
    title = submission.title
    post_text = submission.selftext
    return title, post_text

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        url = request.form.get("url", "")
        if url:
            title, post_text = extract_post_from_url(url)
        else:
            title = request.form.get("title", "")
            post_text = request.form.get("post_text", "")
        text = clean_text(f"{title} {post_text}")
        X_new = vectorizer.transform([text])
        prediction = clf.predict(X_new)[0]
        label = "high-risk" if prediction == 1 else "neutral"
        return render_template("result.html", label=label)
    return render_template("form.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    url = data.get("url", "")
    if url:
        title, post_text = extract_post_from_url(url)
    else:
        title = data.get("title", "")
        post_text = data.get("post_text", "")
    text = clean_text(f"{title} {post_text}")
    X_new = vectorizer.transform([text])
    prediction = clf.predict(X_new)[0]
    label = "high-risk" if prediction == 1 else "neutral"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)




    
