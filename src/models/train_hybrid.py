"""
Loads the trained recommendation models and provide functions to
generate Collaborative Filtering (CF), Content-Based, and Hybrid recomenndations.

- CF recommendations use predicted ratings from the SVD model.
- Content-Based recommendations use movie-to-movie similarity
- Hybrid recommendations combines both CF predictions with content similarity to improve personalization.

This file is used directly by the Flask web application to product recommendations for users.
"""
import pandas as pd
import joblib
from src.config import engine

# Load trained CF model (SVD)
cf_model = joblib.load("src/models/cf_svd.pkl")

# Load movie Dataframe, TF-IDF model, and similarity matrix
content_artifacts = joblib.load("src/models/content_based.pkl")
movies_df = content_artifacts["movies"]
cosine_sim = content_artifacts["cosine_sim"]

# Retrieve a user's rated movies from databse
def get_user_ratings(user_id):
    return pd.read_sql(
        "SELECT movie_id, rating FROM ratings where user_id = :uid",
        engine,
        params={"uid":user_id}
    )

# Predict CF-based movie ratings and return the top recommendations
def get_cf_recommendations(user_id, n=10):
    rated = get_user_ratings(user_id)["movie_id"].tolist()
    candidates = movies_df[~movies_df["movie_id"].isin(rated)].copy()
    
    candidates["pred_rating"] = candidates["movie_id"].apply(
        lambda mid: cf_model.predict(user_id, int(mid)).est
    )

    # Return the top-N highest predicted ratings
    return candidates.sort_values("pred_rating", ascending=False).head(n)

# Return the top-N movies most similar to a given movie (content-based)
def get_content_similiar(movie_id, n=10):
    idx = movies_df.index[movies_df["movie_id"] == movie_id].tolist()
    if not idx:
        return movies_df.head(0)
    
    idx = idx[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]

    movie_indices = [s[0] for s in scores]
    similar = movies_df.iloc[movie_indices].copy()
    similar["similarity"] = [s[1] for s in scores]

    return similar

# Combine CF predictions with content similarity to build the hybrid model
def get_hybrid_recommendations(user_id, n=10):
    cf_candidates = get_cf_recommendations(user_id, n=3*n)

    user_ratings = get_user_ratings(user_id)
    if user_ratings.empty:
        return cf_candidates.head(n)
    
    # Movies similar to user's top-rated items get boosted
    top_liked = user_ratings.sort_values("rating", ascending=False).head(5)
    liked_ids = top_liked["movie_id"].tolist()

    # Create a small boost score for similar movies
    boost = pd.Series(0, index=cf_candidates.index, dtype=float)
    for mid in liked_ids:
        similar = get_content_similiar(mid, n=len(cf_candidates))
        boost_mask = cf_candidates["movie_id"].isin(similar["movie_id"])
        boost[boost_mask] += 0.1

    # Final hybrid score equation
    # CF prediction + content similarity bonus
    cf_candidates["hybrid_score"] = cf_candidates["pred_rating"] + boost

    return cf_candidates.sort_values("hybrid_score", ascending=False).head(n)

def get_trending(n=10):
    """
    Returns top-N movies with the highest number of ratings.
    Used for the 'Trending Now' row.
    """
    query = """
        SELECT
            m.movie_id,
            m.title,
            m.release_year,
            m.genres,
            COUNT(r.rating) AS num_ratings,
            AVG(r.rating)  AS avg_rating
        FROM ratings r
        JOIN movies m ON m.movie_id = r.movie_id
        GROUP BY m.movie_id
        HAVING num_ratings >= 20         -- filter out very rare movies
        ORDER BY num_ratings DESC
        LIMIT :n
    """
    df = pd.read_sql(query, engine, params={"n": n})
    return df

def get_classics(n=10):
    """
    Returns top-N highest rated classic films (before 1980).
    """

    query = """
        SELECT m.movie_id, m.title, m.release_year, m.genres,
               AVG(r.rating) AS avg_rating,
               COUNT(r.rating) AS num_ratings
        FROM ratings r
        JOIN movies m ON m.movie_id = r.movie_id
        WHERE m.release_year < 1980
        GROUP BY m.movie_id
        HAVING num_ratings > 20      -- filter low-sample movies
        ORDER BY avg_rating DESC
        LIMIT :n
    """

    df = pd.read_sql(query, engine, params={"n": n})
    return df

# Because You Watched (Content-based)
from joblib import load

content_model = load("src/models/content_based.pkl")
cosine_sim = content_model["cosine_sim"]
movie_list = content_model["movies"]

def get_because_you_watched(movie_id, n=10):
    idx = movie_list.index[movie_list["movie_id"] == movie_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movie_list.iloc[movie_indices]

def get_last_watched(user_id):
    """Return the most recently rated movie by a user."""
    query = """
        SELECT m.movie_id, m.title, m.release_year
        FROM ratings r
        JOIN movies m ON m.movie_id = r.movie_id
        WHERE r.user_id = :uid
        ORDER BY r.rating_ts DESC
        LIMIT 1
    """
    df = pd.read_sql(query, engine, params={"uid": user_id})
    return df.iloc[0] if not df.empty else None
