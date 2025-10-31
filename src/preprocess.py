import pandas as pd
import os

def main():
    print("🧹 Début du prétraitement des données...")

    # 1️⃣ Définir les chemins
    raw_path = "data/raw"
    processed_path = "data/processed"
    os.makedirs(processed_path, exist_ok=True)

    # 2️⃣ Charger les fichiers
    schedule_file = os.path.join(raw_path, "schedule_sample.csv")
    team_stats_file = os.path.join(raw_path, "team_stats_sample.csv")

    schedule = pd.read_csv(schedule_file)
    team_stats = pd.read_csv(team_stats_file)

    print(f"✅ Fichiers chargés : {len(schedule)} matchs, {len(team_stats)} lignes de stats")

    # 3️⃣ Nettoyage de base du calendrier
    schedule = schedule.dropna(subset=["score"])  # garder uniquement les matchs joués
    schedule["score"] = schedule["score"].astype(str)

    # Extraire les buts domicile / extérieur
    schedule[["home_goals", "away_goals"]] = schedule["score"].str.extract(r"(\d+)–(\d+)")
    schedule["home_goals"] = pd.to_numeric(schedule["home_goals"], errors="coerce")
    schedule["away_goals"] = pd.to_numeric(schedule["away_goals"], errors="coerce")

    # Résultat du match
    def get_result(row):
        if row["home_goals"] > row["away_goals"]:
            return "Home Win"
        elif row["home_goals"] < row["away_goals"]:
            return "Away Win"
        else:
            return "Draw"

    schedule["result"] = schedule.apply(get_result, axis=1)

    # 4️⃣ Supprimer les lignes invalides
    schedule = schedule.dropna(subset=["home_goals", "away_goals"])
    schedule = schedule.drop_duplicates(subset=["date", "home_team", "away_team"])

    # 5️⃣ Sauvegarder le résultat
    output_file = os.path.join(processed_path, "clean_matches.csv")
    schedule.to_csv(output_file, index=False)

    print(f"✅ Fichier nettoyé enregistré dans {output_file}")
    print("🎯 Prétraitement terminé avec succès !")

if __name__ == "__main__":
    main()
