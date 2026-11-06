import pandas as pd
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re
import time

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "transfer_history_2025_26.csv"

SEASON = "2025-26"

LEAGUES = {
    "Premier League": "https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/saison_id/2025",
    "La Liga": "https://www.transfermarkt.com/laliga/transfers/wettbewerb/ES1/saison_id/2025",
    "Bundesliga": "https://www.transfermarkt.com/bundesliga/transfers/wettbewerb/L1/saison_id/2025",
    "Serie A": "https://www.transfermarkt.com/serie-a/transfers/wettbewerb/IT1/saison_id/2025",
    "Ligue 1": "https://www.transfermarkt.com/ligue-1/transfers/wettbewerb/FR1/saison_id/2025",
}


def get_page(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/139.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    return response.text


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def collect_league(league_name, url):

    print("\n" + "=" * 60)
    print(f"Collecting transfers: {league_name}")
    print("=" * 60)

    try:
        html = get_page(url)

    except Exception as error:
        print(f"Could not access {league_name}: {error}")
        return []

    soup = BeautifulSoup(html, "html.parser")

    transfers = []

    # Transfermarkt transfer rows
    rows = soup.select("div.tm-player-transfer-list")

    print(f"Transfer sections found: {len(rows)}")

    for section in rows:

        try:
            player_tag = section.select_one(
                "a.tm-transfer-player-link"
            )

            if not player_tag:
                continue

            player_name = clean_text(
                player_tag.get_text()
            )

            text = clean_text(
                section.get_text(" ", strip=True)
            )

            transfers.append({
                "player_name": player_name,
                "from_club": "",
                "to_club": "",
                "transfer_date": "",
                "transfer_fee": "",
                "transfer_type": text,
                "league": league_name,
                "season": SEASON
            })

        except Exception:
            continue

    return transfers


def main():

    print("=" * 60)
    print("FOOTBALL TRANSFER DATASET BUILDER")
    print("Season:", SEASON)
    print("=" * 60)

    all_transfers = []

    for league_name, url in LEAGUES.items():

        data = collect_league(
            league_name,
            url
        )

        all_transfers.extend(data)

        time.sleep(2)

    if not all_transfers:

        print("\nNo transfers collected.")
        print("The source may have blocked automated requests.")
        return

    df = pd.DataFrame(all_transfers)

    # Remove duplicate records
    df = df.drop_duplicates()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("TRANSFER DATASET COMPLETE")
    print("=" * 60)

    print("Transfers:", len(df))
    print("Saved to:", OUTPUT_FILE)

    print("\nColumns:")
    for column in df.columns:
        print("-", column)


if __name__ == "__main__":
    main()