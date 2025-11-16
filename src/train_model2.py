import os
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# -------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------

def log(msg: str):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

def load_data():
    log("Loading datasets...")

    pre = pd.read_csv("data/processed/model2_preprocessed.csv")
    team_stats = pd.read_csv("data/raw/team_stats_multi_leagues.csv")
    match_stats = pd.read_csv("data/raw/team_match_stats_model2.csv")

    log(f"Preprocessed: {pre.shape}")
    log(f"Team stats: {team_stats.shape}")
    log(f"Match stats: {match_stats.shape}")

    return pre, team_stats, match_stats


# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------

def create_features(pre, team_stats):
    log("Creating team long-term features...")

    selected_cols = [
        "team", "matches_played", "goals_for", "goals_against",
        "matches_home", "goals_for_home", "goals_against_home",
        "matches_away", "goals_for_away", "goals_against_away"
    ]
    team_stats = team_stats[selected_cols]

    ts_home = team_stats.rename(columns=lambda c: f"home_{c}" if c != "team" else "home_team_clean")
    ts_away = team_stats.rename(columns=lambda c: f"away_{c}" if c != "team" else "away_team_clean")

    pre = pre.merge(ts_home, on="home_team_clean", how="left")
    pre = pre.merge(ts_away, on="away_team_clean", how="left")

    pre["goals_for_diff"] = pre["home_goals_for"] - pre["away_goals_for"]
    pre["goals_against_diff"] = pre["home_goals_against"] - pre["away_goals_against"]
    pre["matches_played_diff"] = pre["home_matches_played"] - pre["away_matches_played"]

    return pre


# -------------------------------------------------------------------
# FINAL FEATURES
# -------------------------------------------------------------------

def select_model_features(df):
    log("Selecting features...")

    f = [
        "home_strength", "away_strength", "strength_diff",
        "home_goals_for", "away_goals_for",
        "home_goals_against", "away_goals_against",
        "goals_for_diff", "goals_against_diff",
        "matches_played_diff",
        "home_xg", "away_xg",
    ]

    f = [c for c in f if c in df.columns]

    log(f"Using {len(f)} features")
    return df[f], df["result"]


# -------------------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------------------

def train_xgb(X_train, y_train):
    log("Training XGBoost...")

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    pre, team_stats, match_stats = load_data()

    df = create_features(pre, team_stats)

    # Remove rows without target
    df = df[df["result"].notna()]

    # ------------------------------
    # FIX CLASS LABELS FOR XGBOOST
    # XGBoost expects 0,1,2
    # Your labels = -1,0,1
    # ------------------------------
    df["result_xgb"] = df["result"].replace({
        -1: 0,
        0: 1,
        1: 2
    })

    X, y = select_model_features(df)
    y = df["result_xgb"]

    log("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = train_xgb(X_train, y_train)

    log("Evaluating...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    log(f"Accuracy: {acc:.4f}")
    log(f"F1 Score: {f1:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save_model("models/model2_xgb.json")
    log("Saved model → models/model2_xgb.json")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/model2_training_dataset.csv", index=False)
    log("Saved dataset → data/processed/model2_training_dataset.csv")


if __name__ == "__main__":
    main()
