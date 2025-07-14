# Reddit Suicide Risk Detector
### Overview
This project implements a text classification pipeline to identify suicide risk in Reddit posts. It includes data collection, cleaning, labeling, feature extraction, model training, evaluation, and deployment via a Flask web app. The model uses logistic regression with TF-IDF features.

### Features
- Scrapes posts from specified subreddits using PRAW

 - Cleans and labels data for binary classification

 - Extracts features using TF-IDF vectorization

 - Trains and evaluates a logistic regression model

 - Web interface for risk prediction from Reddit post URLs

### Requirements
 - Python 3.10+

 - pandas

 - scikit-learn

 - nltk

 - praw (Reddit API)

 - Flask

 - joblib

**Install dependencies:**

```
  pip install pandas scikit-learn nltk praw flask joblib
```
**Download NLTK data:**
```
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```
### Usage
**Data Preparation**
Scrape and preprocess Reddit posts, then save as CSV.

**Model Training:**
Train logistic regression on preprocessed data.

**Web Deployment:**
Launch Flask app for real-time inference.

**Prediction:**
Enter a Reddit post URL in the web interface to classify risk.

### File Structure
| File/Folder                  | Purpose                          |
|------------------------------|----------------------------------|
| reddit_posts_preprocessed.csv | Preprocessed dataset             |
| suicide_risk_logreg_model.pkl | Trained logistic regression model|
| tfidf_vectorizer.pkl          | Fitted TF-IDF vectorizer         |
| app.py                        | Flask web application            |
| templates/                    | HTML templates for Flask         |
| static/style.css              | CSS for web interface            |

### Example
Run the Flask app:
```
python app.py
```
Open your browser at http://127.0.0.1:5000/ and submit a Reddit post URL for risk evaluation.

P.S. I tried the same project with deep learning algorithms like Long-Term Short Memory and Convoluted Neural Networks, so I've added those files here as well, but they're not absolutely necessary to download.
