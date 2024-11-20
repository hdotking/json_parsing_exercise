from datetime import datetime
from unittest.mock import patch

import polars as pl

from json_parsing_exercise.exercise_one import (
    ActivityAggregator,
    InstagramData,
    Storage,
)


# (a) Test parsing JSON data
def test_parse_instagram_json():
    raw_json = """{
        "periodStart": "15/02/11",
        "periodEnd": "30/08/21",
        "monthlyPostingDay": 11,
        "comments": [
            ["2/3/21", "Justin Bieber", 5],
            ["5/4/21", "Lady Gaga", 6]
        ]
    }"""

    ig_data = InstagramData.model_validate_json(raw_json)
    assert ig_data.periodStart == datetime(2011, 2, 15)
    assert ig_data.periodEnd == datetime(2021, 8, 30)
    assert len(ig_data.comments) == 2
    assert ig_data.comments[0].user == "Justin Bieber"


# (b) Test storing posts and comments on a daily basis
def test_store_posts_and_comments_daily():
    ig_data = InstagramData.model_validate(
        {
            "periodStart": "01/03/21",
            "periodEnd": "31/03/21",
            "monthlyPostingDay": 11,
            "comments": [
                {"date": "02/03/21", "user": "Justin Bieber", "count": 5},
                {"date": "11/03/21", "user": "Lady Gaga", "count": 6},
            ],
        }
    )

    aggregator = ActivityAggregator(ig_data)
    daily_data = aggregator.initialise_daily_data()

    assert daily_data.shape[0] == 31  # Ensure there are 31 days in March 2021
    assert daily_data["num_comments"].sum() == 11  # Sum of all comments
    post_day = daily_data.filter(pl.col("post_day") == 1).shape[0]
    assert post_day == 1  # There should be 1 post day on the 11th of March


# (c) Test calculating the sum of posts and comments on a daily basis
def test_calculate_daily_sums():
    ig_data = InstagramData.model_validate(
        {
            "periodStart": "01/03/21",
            "periodEnd": "31/03/21",
            "monthlyPostingDay": 11,
            "comments": [
                {"date": "02/03/21", "user": "Justin Bieber", "count": 5},
                {"date": "11/03/21", "user": "Lady Gaga", "count": 6},
            ],
        }
    )

    aggregator = ActivityAggregator(ig_data)
    aggregator.calculate_daily_sums()

    total_activities = aggregator.daily_data_df["total_activities"].sum()
    assert total_activities == 12  # 1 post + 11 comments = 12 activities


# (d) Test calculating monthly aggregate of posts and comments
def test_calculate_monthly_aggregates():
    ig_data = InstagramData.model_validate(
        {
            "periodStart": "01/01/21",
            "periodEnd": "31/03/21",
            "monthlyPostingDay": 11,
            "comments": [
                {"date": "11/01/21", "user": "Justin Bieber", "count": 5},
                {"date": "11/02/21", "user": "Lady Gaga", "count": 6},
                {"date": "11/03/21", "user": "Snoop Dogg", "count": 7},
            ],
        }
    )

    aggregator = ActivityAggregator(ig_data)
    monthly_data = aggregator.calculate_monthly_aggregates()

    assert (
        monthly_data["num_posts"].sum() == 3
    )  # 1 post per month (January, February, March)
    assert monthly_data["num_comments"].sum() == 18  # 5 + 6 + 7 = 18 comments in total


# (e) Test storing monthly totals to a CSV file (with mocking)
@patch("polars.DataFrame.write_csv")
def test_store_monthly_totals_to_csv(mock_write_csv):
    df = pl.DataFrame(
        {
            "month_date": [datetime(2021, 1, 1), datetime(2021, 2, 1)],
            "num_posts": [2, 3],
            "num_comments": [10, 15],
        }
    )

    storage = Storage()
    storage.write_to_csv(df, "IG_user_activity")

    mock_write_csv.assert_called_once_with("IG_user_activity.csv")
