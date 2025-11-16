import soccerdata as sd

from .utils import log, safe_save
from .config import LEAGUES, SEASONS

def extract_team_stats():

    fbref = sd.FBref(leagues=LEAGUES, seasons=SEASONS)

    log("Extracting team match stats...")
    team_match = fbref.read_team_match_stats()

    log("Extracting team season stats...")
    team_season = fbref.read_team_season_stats()

    safe_save(team_match, "data/raw/team_match_stats_model2.csv")
    safe_save(team_season, "data/raw/team_season_stats_model2.csv")

    return team_match, team_season


if __name__ == "__main__":
    extract_team_stats()
