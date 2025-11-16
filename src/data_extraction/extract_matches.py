import soccerdata as sd
from .utils import log, safe_save
from .config import LEAGUES, SEASONS

def extract_matches():

    log("Starting extraction of matches...")

    fbref = sd.FBref(leagues=LEAGUES, seasons=SEASONS)

    log("Reading schedule from FBref...")
    schedule = fbref.read_schedule()

    log(f"Extracted {len(schedule)} matches.")

    safe_save(schedule, "data/raw/schedule_model2.csv")

    return schedule


if __name__ == "__main__":
    extract_matches()
