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

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)

    print("Splitting train and test set")
    # train and test split is 80/20
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    print("Training SVD model")
    algo = SVD(random_state=42)
    algo.fit(trainset)

    print("Evaluating Model")
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"CF model trained! RMSE = {rmse:.4f}")

    joblib.dump(algo, "src/models/cf_svd.pkl")
    print("Model saved to src/models/cf_svd.pkl")

if __name__ == "__main__":
    train_cf_model()
