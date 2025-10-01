import json

import pandas as pd


def segments_and_events_to_long(df):
    rows = []

    for _, row in df.iterrows():
        row_id = row["id"]

        # Parse the segments dict
        if pd.notnull(row["segments"]):
            try:
                segments_dict = json.loads(row["segments"].replace("'", '"'))
                for label, times in segments_dict.items():
                    rows.append(
                        {
                            "original_sample_id": row_id,
                            "original_label": label,
                            "start": times[0],
                            "end": times[1],
                        }
                    )
            except Exception:
                print("Missing segment date")
                print(row_id)

        # Parse the events dict
        if pd.notnull(row["events"]):
            try:
                events_dict = json.loads(row["events"].replace("'", '"'))
                for label, times in events_dict.items():
                    ts = times[0] if isinstance(times, list) else times
                    rows.append(
                        {
                            "original_sample_id": row_id,
                            "original_label": label,
                            "start": ts,
                            "end": ts.replace(
                                "000000", "235959"
                            ),  # end of day for events
                        }
                    )
            except Exception as e:
                print(f"Error parsing events for id {row_id}: {e}")

    # Create new DataFrame from the extracted rows
    return pd.DataFrame(rows)
