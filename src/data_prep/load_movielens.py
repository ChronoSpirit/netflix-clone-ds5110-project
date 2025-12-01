import pandas as pd
from datetime import datetime
from sqlalchemy import text
from src.config import engine

DATA_PATH = "data/raw/"

def create_schema():
    # We drop tables if they exist and then recreate them
    commands = [
    "DROP TABLE IF EXISTS recommendations;",
    "DROP TABLE IF EXISTS ratings;",
    "DROP TABLE IF EXISTS movies;",
    "DROP TABLE IF EXISTS users;",

    """
    CREATE TABLE users (
        user_id INTEGER PRIMARY KEY
    );
    """,

    """
    CREATE TABLE movies (
        movie_id INTEGER PRIMARY KEY,
        title    TEXT NOT NULL,
        release_year INTEGER,
        genres   TEXT
    );
    """,

    """
    CREATE TABLE ratings (
        user_id INTEGER,
        movie_id INTEGER,
        rating REAL,
        rating_ts TEXT,
        PRIMARY KEY (user_id, movie_id),
        FOREIGN KEY (user_id) REFERENCES user(user_id),
        FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
    );
    """,

    """
    CREATE TABLE recommendations (
        user_id INTEGER,
        movie_id INTEGER,
        score REAL,
        algo_type TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, movie_id, algo_type)
    ); 
    """
    ]
    with engine.begin() as conn:
        for command in commands:
            conn.execute(text(command))

def load_movies():
    # Load raw file
    movies = pd.read_csv(
        DATA_PATH + "u.item",
        sep="|",
        header=None,
        encoding="latin-1"
    )

    print("\nLoaded movies shape:", movies.shape)

    # Assign correct column names (24 columns)
    movies.columns = [
        "movie_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    # Extract year from title like "Toy Story (1995)"
    movies["release_year"] = (
        movies["title"]
        .str.extract(r"\((\d{4})\)", expand=False)
        .astype("float")
        .astype("Int64")
    )

    # List genre columns
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    # Create genres string
    movies["genres"] = movies[genre_cols].apply(
        lambda row: "|".join([g for g in genre_cols if row[g] == 1]),
        axis=1
    )

    # Clean selection
    movies_clean = movies[["movie_id", "title", "release_year", "genres"]]

    # Insert into DB
    movies_clean.to_sql("movies", engine, if_exists="append", index=False)

    print(f"Inserted {len(movies_clean)} movies.")


def load_rating_and_users():
    ratings = pd.read_csv(
        DATA_PATH + "u.data",
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    ratings["rating_ts"] = ratings["timestamp"].apply(lambda x: datetime.fromtimestamp(x).isoformat())
    ratings_clean = ratings[["user_id", "movie_id", "rating", "rating_ts"]]

    #user table
    users = ratings_clean[["user_id"]].drop_duplicates()
    users.to_sql("users", engine, if_exists="append", index=False)
    ratings_clean.to_sql("ratings", engine, if_exists="append", index=False)

def main():
    print("Create schema...")
    create_schema()
    print("Loading movies...")
    load_movies()
    print("Loading ratings + users...")
    load_rating_and_users()
    print("Sucess! database stored at netflix_clone.db")

if __name__ == "__main__":
    main()