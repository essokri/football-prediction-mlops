import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier

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
    We take the last known stats for both teams from the training dataset
    """

    # Get last row for each team (contains engineered features)
    home_row = df_train[df_train["home_team_clean"] == home].tail(1)
    away_row = df_train[df_train["away_team_clean"] == away].tail(1)

    if home_row.empty and away_row.empty:
        raise ValueError(f"Aucun historique trouvé pour {home} et {away}.")

    if home_row.empty:
        raise ValueError(f"Aucun historique pour l'équipe domicile : {home}")

    if away_row.empty:
        raise ValueError(f"Aucun historique pour l'équipe extérieure : {away}")

    # We only keep the columns used during training
    feature_cols = [
        "home_strength", "away_strength", "strength_diff",
        "home_goals_for", "away_goals_for",
        "home_goals_against", "away_goals_against",
        "goals_for_diff", "goals_against_diff",
        "matches_played_diff",
        "home_xg", "away_xg",
    ]

    # Build feature row
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
    """
    Model predicts class 0/1/2 corresponding to:
    0 = away win
    1 = draw
    2 = home win
    """

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


if __name__ == "__main__":
    main()
