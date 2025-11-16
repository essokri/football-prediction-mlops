import pandas as pd
import re

def extract_team_from_url(url):
    if pd.isna(url):
        return None
    # Extract last part: "Arsenal-Stats"
    last = url.split("/")[-1]
    # Remove "-Stats"
    team = last.replace("-Stats", "").replace("-", " ")
    return team.strip()

schedule = pd.read_csv("data/raw/schedule_model2.csv")
team_stats = pd.read_csv("data/raw/team_season_stats_model2.csv")

# Extract team name from URL
team_stats["team"] = team_stats["url"].apply(extract_team_from_url)

schedule_teams = set(schedule["home_team"].dropna().unique()) | set(schedule["away_team"].dropna().unique())
team_stat_teams = set(team_stats["team"].dropna().unique())

print("\n=== Teams in schedule.csv ===")
for t in sorted(schedule_teams):
    print(t)

print("\n=== Teams in team_season_stats.csv ===")
for t in sorted(team_stat_teams):
    print(t)
