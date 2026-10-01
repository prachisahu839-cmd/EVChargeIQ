import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("ACN_API_TOKEN")

BASE_URL = "https://ev.caltech.edu/api/v1/sessions/caltech"

all_sessions = []
page = 1

while True:
    print(f"Downloading page {page}...")

    response = requests.get(
        BASE_URL,
        auth=(TOKEN, ""),
        params={"page": page}
    )

    if not response.ok:
        print("API request failed:", response.status_code)
        print(response.text)
        break

    data = response.json()

    sessions = data.get("_items", [])
    all_sessions.extend(sessions)

    print(f"  Sessions downloaded: {len(sessions)}")
    print(f"  Total so far: {len(all_sessions)}")

    links = data.get("_links", {})

    if "next" not in links:
        break

    page += 1

os.makedirs("data/raw", exist_ok=True)

output_file = "data/raw/caltech_sessions.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_sessions, f, indent=2)

print("\n✅ Download complete!")
print(f"Total sessions: {len(all_sessions)}")
print(f"Saved to: {output_file}")