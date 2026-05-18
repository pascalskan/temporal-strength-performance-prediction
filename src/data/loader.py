import pandas as pd


def load_data(path):
    df = pd.read_csv(
        path,
        low_memory=False,
        dtype={"State": "str", "MeetState": "str"}
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df