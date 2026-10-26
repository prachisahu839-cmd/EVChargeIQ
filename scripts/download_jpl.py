import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("ACN_API_TOKEN")

if not TOKEN:
    raise ValueError("ACN_API_TOKEN not found in .env")

BASE_URL = "https://ev.caltech.edu/api/v1/sessions/jpl"

all_sessions = []
page = 1
page_size = 50

session = requests.Session()

while True:
    print(f"Downloading page {page}...")

    success = False

    for attempt in range(1, 6):
        try:
            response = session.get(
                BASE_URL,
                auth=(TOKEN, ""),
                params={
                    "page": page,
                    "max_results": page_size,
                    "pretty": ""
                },
                timeout=120
            )

            response.raise_for_status()
            data = response.json()

            success = True
            break

        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt}/5 failed: {e}")

            if attempt < 5:
                wait_time = attempt * 5
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("  Failed after 5 attempts.")
                raise

    if not success:
        break

    items = data.get("_items", [])

    if not items:
        break

    all_sessions.extend(items)

    print(
        f"  Received: {len(items)} | "
        f"Total: {len(all_sessions)}"
    )

    if len(items) < page_size:
        break

    page += 1

    # Small delay to avoid sending requests too quickly
    time.sleep(1)

output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "jpl_sessions.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_sessions, f, indent=2)

print("\nDownload complete.")
print(f"Total JPL sessions: {len(all_sessions)}")
print(f"Saved to: {output_file}")