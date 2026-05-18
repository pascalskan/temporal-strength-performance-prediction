import pandas as pd

SOURCE = "data/openpowerlifting.csv"
TARGET = "data/test_fixture.csv"

RAW_MALE = 75
RAW_FEMALE = 75
WRAPS_MALE = 40
WRAPS_FEMALE = 40

MIN_COMPETITIONS = 3
SEED = 42


def sample_group(df, equipment_mask, sex_value, n):
    subset = df[equipment_mask & (df["Sex"] == sex_value)].copy()

    eligible = (
        subset.groupby("Name")
        .size()
        .reset_index(name="count")
    )

    eligible = eligible[eligible["count"] >= MIN_COMPETITIONS]

    athletes = eligible["Name"].sample(
        n=min(n, len(eligible)),
        random_state=SEED
    )

    return subset[subset["Name"].isin(athletes)]


df = pd.read_csv(SOURCE)

raw_mask = df["Equipment"] == "Raw"
wraps_mask = df["Equipment"] != "Raw"

fixture_parts = [
    sample_group(df, raw_mask, "M", RAW_MALE),
    sample_group(df, raw_mask, "F", RAW_FEMALE),
    sample_group(df, wraps_mask, "M", WRAPS_MALE),
    sample_group(df, wraps_mask, "F", WRAPS_FEMALE),
]

fixture = pd.concat(fixture_parts, ignore_index=True)

fixture = fixture.sort_values(["Name", "Date"])

fixture.to_csv(TARGET, index=False)

print("Fixture created.")
print(f"Rows: {len(fixture)}")
print(f"Athletes: {fixture['Name'].nunique()}")

print("\nBreakdown:")
print(fixture.groupby(['Equipment', 'Sex'])['Name'].nunique())