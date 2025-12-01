"""
Loads the trained recommendation models and provide functions to
generate Collaborative Filtering (CF), Content-Based, and Hybrid recomenndations.

- CF recommendations use predicted ratings from the SVD model.
- Content-Based recommendations use movie-to-movie similarity
- Hybrid recommendations combines both CF predictions with content similarity to improve personalization.

This file is sued directly by the Flask web application to product recommendations for users.
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
    return pd.real_sql(
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