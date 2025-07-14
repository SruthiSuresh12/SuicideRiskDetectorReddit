import nest_asyncio
nest_asyncio.apply()

import asyncpraw
import pandas as pd
import asyncio

async def fetch_posts():
    reddit = asyncpraw.Reddit(
        client_id="2eX3T9gzL07MzGepVqB_-A",
        client_secret="VS-2e-FyNOoXKMC5YttUOe4CJHiw0A",
        user_agent="DepressionWatcher"
    )

    subreddits = [
        "SuicideWatch", "depression", "anxiety", "mentalhealth", "mentalillness", "BPD", "bipolar",
        "autism", "MentalHealthUK", "socialanxiety", "talktherapy", "askatherapist", "offmychest",
        "traumatoolbox", "dbtselfhelp", "bodyacceptance", "MMFB", "mentalhealthmemes", "anxietymemes",
        "MentalHealthSupport", "malementalhealth", "mentalhealthph", "mentalhealthsupport",
        "mentalhealthbabies", "emotionalsupport", "helpme", "Advice", "KindVoice", "Vent", "venting",
        "Feels", "sad", "CasualConversation", "MomForAMinute", "DadForAMinute", "BenignExistence",
        "findafriend", "relationship_advice", "internetparents", "freecompliments", "Confessions",
        "Offmychest", "AskReddit", "TodayILearned", "pics", "funny", "worldnews", "science", "movies",
        "books", "technology", "gaming", "sports", "Music", "Art", "food", "DIY", "space", "History",
        "television", "Documentaries", "InternetIsBeautiful", "travel", "photography", "cooking",
        "gardening", "Fitness", "cars", "Bicycling", "boardgames", "CampingandHiking", "coffee", "tea",
        "knitting", "woodworking"
    ]

    posts = []

    for subreddit_name in subreddits:
        try:
            subreddit = await reddit.subreddit(subreddit_name)
            async for submission in subreddit.new(limit=500):
                posts.append({
                    "post_id": submission.id,
                    "subreddit": subreddit_name,
                    "timestamp": submission.created_utc,
                    "title": submission.title,
                    "post_text": submission.selftext,
                    "user_id": submission.author.name if submission.author else None,
                    "upvotes": submission.score,
                    "comments": submission.num_comments
                })
        except Exception:
            continue

    df = pd.DataFrame(posts)
    df.to_csv("reddit_posts.csv", index=False)

loop = asyncio.get_event_loop()
loop.run_until_complete(fetch_posts())
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df = pd.read_csv("reddit_posts_labeled.csv")
df["clean_text"] = df["title"].fillna("") + " " + df["post_text"].fillna("")
df["clean_text"] = df["clean_text"].apply(clean_text)
df.to_csv("reddit_posts_preprocessed.csv", index=False)
