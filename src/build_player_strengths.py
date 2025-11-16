import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def log(msg):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now}] {msg}")


def detect_numeric_cols(df):
    """Detect valid numerical columns automatically."""
    numeric_cols = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors="raise")
            numeric_cols.append(col)
        except:
            pass
    return numeric_cols


def compute_player_scores(df):
    log("Computing scores...")

    df = df.copy()
    df = df.fillna(0)

    # Detect numeric stats
    numeric_cols = detect_numeric_cols(df)

    # Simple numeric-based score: sum of normalized stats
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df["player_score"] = df[numeric_cols].sum(axis=1)

    return df[["team", "player", "pos", "player_score"]]


def main():
    log("Loading player stats...")

    df = pd.read_csv("data/raw/player_season_stats_model2.csv")

    # Fix team names
    log("Fixing team names...")
    df["team"] = df["team"].astype(str).str.strip()

    df_scores = compute_player_scores(df)

    out_path = "data/processed/player_strengths.csv"
    df_scores.to_csv(out_path, index=False)

    log(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
