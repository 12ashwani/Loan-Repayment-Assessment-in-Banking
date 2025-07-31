import json
import glob

for file in glob.glob("*.ipynb"):
    with open(file, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if "widgets" in nb.get("metadata", {}):
        if "state" not in nb["metadata"]["widgets"]:
            print(f"Fixing '{file}'...")
            nb["metadata"]["widgets"]["state"] = {}

    with open(file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
