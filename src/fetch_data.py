import soccerdata as sd
import pandas as pd
import os

def main():
    # 1️⃣ Créer le dossier où on va sauvegarder les données
    os.makedirs("data/raw", exist_ok=True)
    print("📥 Début de la collecte de données test (1 ligue)...")

    # 2️⃣ Télécharger les données d'une seule ligue (ex: Premier League)
    fbref = sd.FBref(leagues=["ENG-Premier League"], seasons=["2324"])

    # 3️⃣ Lire le calendrier (schedule)
    schedule = fbref.read_schedule()
    schedule.to_csv("data/raw/schedule_sample.csv", index=False)
    print(f"✅ Schedule téléchargé : {schedule.shape[0]} matchs")

    # 4️⃣ Lire les statistiques d’équipe
    team_stats = fbref.read_team_season_stats(stat_type="standard")
    team_stats.to_csv("data/raw/team_stats_sample.csv", index=False)
    print(f"✅ Statistiques d'équipes téléchargées : {team_stats.shape[0]} lignes")

    print("🎯 Collecte test terminée avec succès !")

if __name__ == "__main__":
    main()