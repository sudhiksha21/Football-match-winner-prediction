from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model & helpers
model = joblib.load("model/random_forest.pkl")
features = joblib.load("model/features.pkl")
team_le = joblib.load("model/team_encoder.pkl")
opp_le = joblib.load("model/opponent_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    teams = sorted(team_le.classes_)

    if request.method == "POST":
        team = request.form["team"]
        opponent = request.form["opponent"]
        venue = request.form["venue"]

        team_enc = team_le.transform([team])[0]
        opp_enc = opp_le.transform([opponent])[0]
        venue_enc = 1 if venue == "Home" else 0

        # Simple baseline inputs for demo
        input_data = np.array([[
            team_enc,
            opp_enc,
            venue_enc,
            1.2,  # gf_avg_5
            1.1,  # ga_avg_5
            0.5,  # win_rate_5
            1.4,  # xg
            1.2,  # xga
            52    # possession
        ]])

        probs = model.predict_proba(input_data)[0]
        classes = model.classes_

        idx = np.argmax(probs)
        result_map = {1: "Win", 0: "Draw", -1: "Loss"}

        prediction = result_map[classes[idx]]
        confidence = round(probs[idx] * 100, 2)

    return render_template(
        "index.html",
        teams=teams,
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
