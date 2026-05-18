BASELINE_FEATURES = [
    "Sex",
    "Age",
    "BodyweightKg"
]

ENGINEERED_FEATURES = [
    "Sex",
    "Age",
    "BodyweightKg",
    "Prev_Total",
    "Rolling_Mean_3",
    "Rolling_Std_3",
    "Comp_Count",
    "Days_Since_Last",
    "PB",
    "Peak_Distance",
    "Improvement",
    "Momentum",
    "CV",
    "Career_Length_Days",
    "Experience_Density",
    "Age_Squared"
]

ATTEMPT_COLUMNS = [
    "Squat1Kg",
    "Squat2Kg",
    "Squat3Kg",
    "Bench1Kg",
    "Bench2Kg",
    "Bench3Kg",
    "Deadlift1Kg",
    "Deadlift2Kg",
    "Deadlift3Kg"
]

LEAKAGE_COLUMNS = [
    "Success_Rate",
    "Aggression",
    
    "Clutch",
    "Risk"
]