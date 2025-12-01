"""
Script builds the Content-Based Filtering model for the recommendation system.

Content-Based Filtering recommends movies based on descriptive features like
movie titles and genres, rather than the user's behavior. 
"""
import joblib
import pandas as pd
from sqlalchemy import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import engine

def build_content_model():
    # Loads movie_id, title and genres from database
    print("Loading movies")
    movies = pd.read_sql("SELECT movie_id, title, genres FROM movies", engine)

    # Combine title and genres into one text field for TF-IDF processing
    print("Building the text features")
    movies["text"] = (movies["title"].fillna("") + " " +
                      movies["genres"].fillna(" "))
    # Convert the movie text into TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["text"])

    # Compute similarity between all movie pairs using cosine similarity
    print("Computing cosine similiarity")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Saves all important objects for recommendations
    joblib.dump(
        {"movies":movies, "tfidf":tfidf, "cosine_sim": cosine_sim},
        "src/models/content_based.pkl"
    )

    print("Content-based model Saved.")

if __name__ == "__main__":
    build_content_model()