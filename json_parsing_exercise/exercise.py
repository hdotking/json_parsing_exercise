import json
from datetime import datetime
from typing import List, Optional

import polars as pl
from pydantic import BaseModel, ValidationError, field_validator

"""
This module uses three classes to parse, validate, and store Instagram data. Custom validation logic with
field_validator is employed due to the non-standard date format and conversion of lists to Comment objects.
While @classmethod is used for clarity, it isn't technically required, as mode=before implies that validation
occurs before model instantiation.

The separation of ActivityAggregator and Storage classes promotes modularity and testability, adhering to the
Single Responsibility Principle. For extending to multiple platforms, a base class approach enables compliance
with the Open/Closed Principle.

E.g.

class SocialMediaData(BaseModel):
    periodStart: datetime
    periodEnd: datetime
    ...

class InstagramData(SocialMediaData):
    comments: List[Comment]
    ...
"""


class Comment(BaseModel):
    date: datetime
    user: str
    count: int

    @field_validator("date", mode="before")
    @classmethod
    def validate_dates(cls, v):
        try:
            return datetime.strptime(v, "%d/%m/%y")
        except ValueError:
            raise ValueError(f"Invalid date format: {v}")


class InstagramData(BaseModel):
    periodStart: datetime
    periodEnd: datetime
    monthlyPostingDay: int
    comments: List[Comment]

    @field_validator("periodStart", "periodEnd", mode="before")
    @classmethod
    def validate_dates(cls, v):
        try:
            return datetime.strptime(v, "%d/%m/%y")
        except ValueError:
            raise ValueError(f"Invalid date format: {v}")

    @field_validator("monthlyPostingDay")
    @classmethod
    def validate_day(cls, v):
        if not (1 <= v <= 31):
            raise ValueError("Monthly posting day must be between 1 and 31")
        return v

    @field_validator("comments", mode="before")
    @classmethod
    def convert_comments(cls, v):  # Convert list of lists to list of Comment objects
        if isinstance(v, list) and all(isinstance(i, list) for i in v):
            return [{"date": item[0], "user": item[1], "count": item[2]} for item in v]
        return v


class ActivityAggregator:
    def __init__(self, ig_data):
        self.ig_data: InstagramData = ig_data
        self.daily_data_df: pl.Dataframe = self.initialise_daily_data()
        self.monthly_data_df: Optional[pl.Dataframe] = None

    def initialise_daily_data(self) -> pl.DataFrame:
        # Create a polars DataFrame with one row for each date between periodStart and periodEnd
        dates = pl.date_range(
            self.ig_data.periodStart, self.ig_data.periodEnd, eager=True
        )
        df = pl.DataFrame({"date": dates})

        # Create the "post_day" column which is 1 on a post day and 0 otherwise
        df = df.with_columns(
            pl.when(pl.col("date").dt.day() == self.ig_data.monthlyPostingDay)
            .then(1)
            .otherwise(0)
            .alias("post_day")
        )

        df = df.with_columns(pl.lit(0).alias("num_comments"))

        # Convert comments to a DataFrame and ensure the 'date' column is of the same type as in 'df'
        comments_df = pl.DataFrame(
            {
                "date": [c.date for c in self.ig_data.comments],
                "count": [c.count for c in self.ig_data.comments],
            }
        ).with_columns(pl.col("date").cast(pl.Date))

        # Join the DataFrames on the 'date' column
        df = (
            df.join(comments_df, on="date", how="left")
            .with_columns(pl.col("count").fill_null(0))
            .with_columns(
                (pl.col("num_comments") + pl.col("count")).alias("num_comments")
            )
            .drop("count")
        )
        return df

    def calculate_daily_sums(self) -> None:
        self.daily_data_df = self.daily_data_df.with_columns(
            (pl.col("post_day") + pl.col("num_comments")).alias("total_activities")
        )

    def calculate_monthly_aggregates(self) -> pl.DataFrame:
        """
        calculate the aggregate number of posts and comments on a monthly basis
        """
        # Group by month and sum the post_day, num_comments, and total_activities columns
        monthly_aggregate_df = self.daily_data_df.group_by_dynamic(
            "date", every="1mo"
        ).agg(
            [
                pl.col("post_day").sum().alias("num_posts"),
                pl.col("num_comments").sum(),
            ]
        )
        monthly_aggregate_df = monthly_aggregate_df.rename({"date": "month_date"})

        # Assign the monthly aggregation to self.monthly_data_df
        self.monthly_data_df = monthly_aggregate_df
        return monthly_aggregate_df


class Storage:
    # (e) Store monthly totals to CSV
    def write_to_csv(self, df: pl.DataFrame, filename: str):
        df.write_csv(f"{filename}.csv")


if __name__ == "__main__":
    # (a) parse the json script (below) from the scraper
    raw_ig_json = """{
        "periodStart": "15/02/11",
        "periodEnd": "30/08/21",
        "monthlyPostingDay": 11,
        "comments": [
            ["2/3/21", "Justin Bieber", 5],
            ["5/4/21", "Lady Gaga", 6],
            ["5/4/21", "Snoop Dog", 2],
            ["13/5/21", "Justin Bieber", 3]
        ]
    }"""
    raw_ig_dict = json.loads(raw_ig_json)

    try:
        data = InstagramData.model_validate(raw_ig_dict)
    except ValidationError as e:
        print("Validation failed:", e)

    # (b) store the number of (i) posts and (ii)comments on a daily basis across the time period in a python class
    ig_aggregator = ActivityAggregator(data)
    print(f"{ig_aggregator.daily_data_df}")

    # (c) calculate the sum of posts and comments on a daily basis
    ig_aggregator.calculate_daily_sums()
    print(f"{ig_aggregator.daily_data_df}")

    # (d) calculate the aggregate number of posts and comments on a monthly basis
    monthly_data = ig_aggregator.calculate_monthly_aggregates()
    print(f"{monthly_data}")

    # (e) store the monthly totals for the whole period for (i) posts and (ii) comments in a csv file
    storage = Storage()
    storage.write_to_csv(
        df=monthly_data,
        filename="IG_user_activity",
    )
