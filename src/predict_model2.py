import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
import os

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# UTILS
# ----------------------------------------------------------

def log(msg):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")


# ----------------------------------------------------------
# LOAD MODEL + DATA
# ----------------------------------------------------------

def load_artifacts():
    log("Loading model & datasets...")

    model = XGBClassifier()
    model.load_model("models/model2_xgb.json")

    df_train = pd.read_csv("data/processed/model2_training_dataset.csv")

    return model, df_train


# ----------------------------------------------------------
# BUILD FEATURE ROW FOR PREDICTION
# ----------------------------------------------------------

def build_input_features(df_train, home, away):
    """
    We take the latest known stats for both teams
    """

    home_row = df_train[df_train["home_team_clean"] == home].tail(1)
    away_row = df_train[df_train["away_team_clean"] == away].tail(1)

    if home_row.empty:
        raise ValueError(f"No history found for HOME team: {home}")
    if away_row.empty:
        raise ValueError(f"No history found for AWAY team: {away}")

    # Build the row
    row = pd.DataFrame({
        "home_strength": home_row["home_strength"].values[0],
        "away_strength": away_row["away_strength"].values[0],
        "strength_diff": home_row["home_strength"].values[0] - away_row["away_strength"].values[0],

        "home_goals_for": home_row["home_goals_for"].values[0],
        "away_goals_for": away_row["away_goals_for"].values[0],

        "home_goals_against": home_row["home_goals_against"].values[0],
        "away_goals_against": away_row["away_goals_against"].values[0],

        "goals_for_diff": home_row["home_goals_for"].values[0] - away_row["away_goals_for"].values[0],
        "goals_against_diff": home_row["home_goals_against"].values[0] - away_row["away_goals_against"].values[0],

        "matches_played_diff": home_row["home_matches_played"].values[0] - away_row["away_matches_played"].values[0],

        "home_xg": home_row["home_xg"].values[0],
        "away_xg": away_row["away_xg"].values[0],
    }, index=[0])

    return row


# ----------------------------------------------------------
# PREDICT
# ----------------------------------------------------------

def predict(model, features, home, away):
    proba = model.predict_proba(features)[0]
    pred = np.argmax(proba)

    mapping = {
        0: f"{away} WIN",
        1: "DRAW",
        2: f"{home} WIN"
    }

    return mapping[pred], proba


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    model, df_train = load_artifacts()

    print("\n=== FOOTBALL MATCH PREDICTION ===\n")
    home = input("Home team: ").strip()
    away = input("Away team: ").strip()

    features = build_input_features(df_train, home, away)

    outcome, proba = predict(model, features, home, away)

    print("\n================= RESULT =================")
    print(f"Prediction: {outcome}")
    print(f"Probas:")
    print(f"  Away Win : {proba[0]:.3f}")
    print(f"  Draw     : {proba[1]:.3f}")
    print(f"  Home Win : {proba[2]:.3f}")
    print("==========================================\n")

    # ----------------------------------------------------------
    # DVC OUTPUT SAVE
    # ----------------------------------------------------------
    os.makedirs("data/predictions", exist_ok=True)
    path = "data/predictions/model2_predictions.csv"

    pd.DataFrame([{
        "home_team": home,
        "away_team": away,
        "prediction": outcome,
        "proba_away_win": proba[0],
        "proba_draw": proba[1],
        "proba_home_win": proba[2],
    }]).to_csv(path, index=False)

    print(f"📝 Saved to {path}")


if __name__ == "__main__":
    main()
