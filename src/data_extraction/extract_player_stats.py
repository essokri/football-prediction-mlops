import soccerdata as sd
from .utils import log, safe_save
from .config import LEAGUES, SEASONS

def extract_player_stats():

    log("Starting extraction of player SEASON stats...")

    # Charger FBref pour les ligues et saisons sélectionnées
    fbref = sd.FBref(leagues=LEAGUES, seasons=SEASONS)

    # Extraction directe des stats cumulées des joueurs
    log("Reading player season stats...")
    player_season_stats = fbref.read_player_season_stats()
    # Reset index pour transformer Player / Team / League / Season en colonnes
    player_season_stats = player_season_stats.reset_index()


    # Log du résultat
    log(f"Extracted {len(player_season_stats)} player season records.")

    # Sauvegarde
    safe_save(player_season_stats, "data/raw/player_season_stats_model2.csv")

    return player_season_stats


if __name__ == "__main__":
    extract_player_stats()
