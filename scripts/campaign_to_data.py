import json
from pathlib import Path

from utils import campaign_to_tables, validate_campaign

if __name__ == "__main__":
    with Path("labels/disfor.geojson").open("r") as f:
        campaign = json.load(f)
    issues = validate_campaign(campaign)
    if issues:
        print(issues)
        print(",".join([i.split(" ")[1][:-1] for i in issues]))
    else:
        s, l = campaign_to_tables(campaign)
        s.to_parquet("data/samples.parquet")
        l.to_parquet("data/labels.parquet")
