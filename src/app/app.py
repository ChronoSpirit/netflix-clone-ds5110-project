from flask import Flask, render_template, request
from src.models.train_hybrid import (
    get_cf_recommendations, 
    get_hybrid_recommendations,
    get_trending,
    get_classics,
    get_because_you_watched
    )
from src.config import engine
import pandas as pd

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/", methods=["GET"])
def index():
    users = pd.read_sql("SELECT DISTINCT user_id FROM ratings ORDER BY user_id LIMIT 20", engine)
    user_ids = users["user_id"].tolist()
    return render_template("index.html", user_ids=user_ids)

@app.route("/recommendations", methods=["POST"])
def recommendations():
    user_id = int(request.form["user_id"])
    algo = request.form.get("algo", "cf")

    if algo == "hybrid":
        recs = get_hybrid_recommendations(user_id,n=10)
    else:
        recs = get_cf_recommendations(user_id,n=10)

    # Additional rows
    trending = get_trending(10).to_dict(orient="records")   
    classics = get_classics(10).to_dict(orient="records")

    # Use the first recommended movie for "Because You Watched"
    if len(recs) > 0:
        selected_movie_id = recs.iloc[0]["movie_id"]
        because = get_because_you_watched(selected_movie_id, 10).to_dict(orient="records")
    else:
        because = []

    return render_template(
        "recommendations.html",
        user_id=user_id,
        algo=algo,
        main_recs=recs.to_dict(orient="records"),
        trending=trending,
        classics=classics,
        because=because,
    )

if __name__ == "__main__":
    app.run(debug=True)
