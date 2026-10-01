import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(
        BASE,
        "data",
        "world_cup_2026_team_prematch_features.csv"
    )
)

original = [
    "avg_goals_scored_before",
    "opponent_avg_goals_conceded_before",
    "scoring_match_rate_before",
    "opponent_clean_sheet_rate_before",
    "team_fifa_points",
    "opponent_fifa_points",
    "scoring_consistency_before",
    "opponent_points_per_match_before"
]

alternative = [
    "avg_goals_scored_before",
    "opponent_avg_goals_conceded_before",
    "team_fifa_points",
    "opponent_fifa_points",
    "win_rate_before",
    "opponent_scoring_match_rate_before",
    "matches_played_before",
    "opponent_clean_sheet_rate_before"
]

# Split by MATCH, keeping both team observations together.
match_ids = df["match_id"].unique()

train_ids, test_ids = train_test_split(
    match_ids,
    test_size=0.2,
    random_state=42
)

train = df[df["match_id"].isin(train_ids)]
test = df[df["match_id"].isin(test_ids)]

for name, predictors in [
    ("Original", original),
    ("Alternative", alternative)
]:
    model = LinearRegression()

    model.fit(
        train[predictors],
        train["goals_scored"]
    )

    predictions = model.predict(test[predictors])

    r2 = r2_score(
        test["goals_scored"],
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["goals_scored"],
            predictions
        )
    )

    # Normalization uses the training-target standard deviation.
    train_std = train["goals_scored"].std(ddof=0)
    nrmse = rmse / train_std

    print(f"\n{name} model")
    print(f"Train observations: {len(train)}")
    print(f"Test observations: {len(test)}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Normalized RMSE: {nrmse:.4f}")