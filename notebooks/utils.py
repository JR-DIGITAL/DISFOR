import json
from datetime import datetime

import pandas as pd
import polars as pl

EVENT_LABELS = {"5", "6", "7", "8", "9", "a", "c", "d", "e"}


def pickle_to_json_flags(meta_df):
    import pickle
    from pathlib import Path

    # reformat windthrow pickles to new flag format
    pickle_paths = Path(
        r"\\digs110\FER\HR-VPP2\Calibration\Interpretation\windthrow\Flags"
    ).glob("*.pickle")
    for p in pickle_paths:
        with p.open("rb") as f:
            sample_id = p.stem.split("_")[1]
            row = meta_df.query(f"id == {sample_id}").iloc[0]
            sample = {
                "flags": dict(
                    sorted(
                        {
                            k.strftime("%Y-%m-%d %H:%M:%S.%f"): v
                            for k, v in pickle.load(f).items()
                        }.items()
                    )
                ),
                "confidence": row["confidence"].lower(),
                "comment": row["comment"],
                "interpreter": row["interpret1"],
            }
        with open(f"../data/windthrow_flags/flags_{sample_id}.json", "w") as f:
            json.dump(sample, f, indent=4)


def check_flag_integrity(flag_paths):
    bad_ids = []
    bad = 0
    for flag_path in flag_paths:
        sample_id = flag_path.stem.split("_")[-1]
        with flag_path.open("r") as f:
            sample = json.load(f)
        segments = {}
        sample["flags"] = dict(sorted(sample["flags"].items()))
        iterator = iter([(k, v) for k, v in sample["flags"].items() if v != "d"])
        previous_label = None
        for flag in iterator:
            if flag == "d":
                continue
            time_str, label = flag
            time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.000000")
            if label in EVENT_LABELS:
                if previous_label == label:
                    print("Wrong Event")
                    print(sample_id)
                    print(json.dumps(sample, indent=4))
                    bad += 1
                    bad_ids.append(sample_id)
                    break
                previous_label = label
                if segments.get(label) is None:
                    segments[label] = []
                segments[label].extend(
                    [time, time.replace(hour=23, minute=59, second=59)]
                )
            else:
                previous_label = label
                end_segment = next(iterator, (None, None))
                if end_segment[1] != label:
                    print("Missing Segment")
                    print(sample_id)
                    print(json.dumps(sample, indent=4))
                    bad += 1
                    bad_ids.append(sample_id)
                    break
                if segments.get(label) is None:
                    segments[label] = []
                segments[label].append(time)
                segments[label].sort()


def flags_to_label_df(flag_paths):
    bad = 0
    bad_ids = []
    rows = []
    for flag_path in flag_paths:
        sample_id = flag_path.stem.split("_")[-1]
        with flag_path.open("r") as f:
            sample = json.load(f)
        sample["flags"] = dict(sorted(sample["flags"].items()))
        flag_list = [(k, v) for k, v in sample["flags"].items() if v != "d"]
        iterator = iter(flag_list)
        for i, (time_str, label) in enumerate(iterator):
            time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.000000")
            if label in EVENT_LABELS:
                # handle 9 (abiotic other, mostly used as drought, i.e. segment)
                if label == "9" and i + 1 < len(flag_list) and flag_list[i + 1] == 9:
                    end_time_str, label = next(iterator)
                    end_time = datetime.strptime(
                        end_time_str, "%Y-%m-%d %H:%M:%S.000000"
                    )
                else:
                    end_time = time.replace(hour=23, minute=59, second=59)
                rows.append(
                    {
                        "original_label": label,
                        "start": time,
                        "end": end_time,
                        "original_sample_id": sample_id,
                    }
                )
            else:
                end_time_str, end_label = next(iterator, (None, None))
                # Special handling of c class: c is non tree vegetation removal. The tree segment does not get interrupted,
                # (since the canopy did not change)
                while end_label == "c":
                    rows.append(
                        {
                            "original_label": end_label,
                            "start": datetime.strptime(
                                end_time_str, "%Y-%m-%d %H:%M:%S.000000"
                            ),
                            "end": datetime.strptime(
                                end_time_str, "%Y-%m-%d %H:%M:%S.000000"
                            ).replace(hour=23, minute=59, second=59),
                            "original_sample_id": sample_id,
                        }
                    )
                    end_time_str, end_label = next(iterator, (None, None))
                if end_label != label:
                    print("Missing Segment")
                    print(sample_id)
                    print(json.dumps(sample, indent=4))
                    bad += 1
                    bad_ids.append(sample_id)
                    break

                end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S.000000")

                rows.append(
                    {
                        "original_label": label,
                        "start": time,
                        "end": end_time,
                        "original_sample_id": sample_id,
                    }
                )
    return pl.DataFrame(rows)


def flags_to_sample_df(flag_paths):
    rows = []
    for flag_path in flag_paths:
        sample_id = flag_path.stem.split("_")[-1]
        with flag_path.open("r") as f:
            sample = json.load(f)
        del sample["flags"]
        sample["original_sample_id"] = sample_id
        rows.append(sample)
    return pl.DataFrame(rows)


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
