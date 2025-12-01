import joblib
import pandas as pd
from sqlalchemy import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import engine

def build_content_model():
    print("Loading movies")
    movies = pd.read_sql("SELECT movie_id, title, genres FROM movies", engine)

    print("Building the text features")
    movies["text"] = (movies["title"].fillna("") + " " +
                      movies["genres"].fillna(" "))
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["text"])

    print("Computing cosine similiarity")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    joblib.dump(
        {"movies":movies, "tfidf":tfidf, "cosine_sim": cosine_sim},
        "src/models/content_based.pkl"
    )

    print("Content-based model Saved.")

if __name__ == "__main__":
    build_content_model()