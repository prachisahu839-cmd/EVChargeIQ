import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("ACN_API_TOKEN")

url = "https://ev.caltech.edu/api/v1/sessions/caltech"

response = requests.get(
    url,
    auth=(TOKEN, ""),
    params={"pretty": ""}
)

print("Status code:", response.status_code)

if response.ok:
    data = response.json()

    print("✅ API connection successful!")
    print("Sessions on this page:", len(data.get("_items", [])))
    print("Metadata:", data.get("_meta"))

else:
    print("❌ API request failed")
    print(response.text)