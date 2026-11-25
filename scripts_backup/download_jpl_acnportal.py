
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from acnportal.acndata import DataClient

load_dotenv()

token = os.getenv("ACN_API_TOKEN")
if not token:
    raise ValueError("ACN_API_TOKEN not found in .env")

output = Path("data/raw/jpl_sessions.json")
output.parent.mkdir(parents=True, exist_ok=True)

# Resume from saved records if a previous run was interrupted.
sessions_by_id = {}

if output.exists():
    with output.open("r", encoding="utf-8") as f:
        for session in json.load(f):
            sessions_by_id[session["sessionID"]] = session
    print(f"Existing sessions loaded: {len(sessions_by_id)}")

client = DataClient(token)

try:
    # Fetch sessions from the official ACN Python client.
    for session in client.get_sessions("jpl"):
        sid = session.get("sessionID")
        if sid:
            sessions_by_id[sid] = session

        # Save progress every 100 new/processed records.
        if len(sessions_by_id) % 100 == 0 and sessions_by_id:
            with output.open("w", encoding="utf-8") as f:
                json.dump(
                    list(sessions_by_id.values()),
                    f,
                    indent=2,
                    default=str
                )
            print(f"Saved {len(sessions_by_id)} unique sessions")
            time.sleep(0.2)

except Exception as e:
    print(f"\nDownload interrupted: {type(e).__name__}: {e}")
    print("Saving collected sessions before exiting...")

finally:
    with output.open("w", encoding="utf-8") as f:
        json.dump(
            list(sessions_by_id.values()),
            f,
            indent=2,
            default=str
        )

print(f"\nSessions saved: {len(sessions_by_id)}")
print(f"File: {output}")

