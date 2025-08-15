import json

with open("./data/old_preprocessors/metadata.json", "r") as f:
    events = json.load(f)["events"]
    print(len(events))