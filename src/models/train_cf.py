"""
Script that trains the Collaborative Filtering (CF) recommendation model.
Uses Surprise library's SVD which is a matrix factorization algorithm

CF model predicts how a user would rate a mobie based off of patterns that is learned
from all user-movie interactions in the MovieLens dataset.
"""
import joblib
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from src.config import engine

def load_ratings_df():
    df = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", engine)
    return df

def train_cf_model():
    print("Loading ratings...")
    df = load_ratings_df()

    # Surprise library's reader to interpret the rating scale
    reader = Reader(rating_scale=(1, 5))
    # Convert the DF to a Surprise Dataset
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)

    print("Splitting train and test set")
    # train and test split is 80/20
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # SVD = Matrix Factorization algorithm used by the Netflix Prize solutions
    print("Training SVD model")
    algo = SVD(random_state=42)
    # Train model on training set
    algo.fit(trainset)

    # Evaluating the model performance with RMSE
    print("Evaluating Model")
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"CF model trained! RMSE = {rmse:.4f}") # Should spit out ~0.93 = 93% which is good!

    # Save the train CF model for the recommendation generation
    joblib.dump(algo, "src/models/cf_svd.pkl")
    print("Model saved to src/models/cf_svd.pkl")

if __name__ == "__main__":
    train_cf_model()
