import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load preprocessed data
df = pd.read_csv("reddit_posts_preprocessed.csv")
texts = df["clean_text"].astype(str).tolist()
labels = df["label"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
maxlen = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding="post")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=maxlen),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    X_train_pad, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

from sklearn.metrics import classification_report

y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
print(classification_report(y_test, y_pred, target_names=["neutral", "high-risk"]))
model.save("suicide_risk_lstm_model.keras")
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)



