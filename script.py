import polars as pl
import sys
import time
from datetime import datetime

if __name__ == "__main__":
    start = sys.argv[1] + " " + sys.argv[2]
    end = sys.argv[3] + " " + sys.argv[4]

    if start >= end:
        print("Start date must be before end date")
        exit()

    start = datetime.strptime(start, "%Y-%m-%d %H")
    end = datetime.strptime(end, "%Y-%m-%d %H")

    start_time = time.perf_counter_ns()

    ranks = (
        pl.scan_parquet("2022pyarrow.parquet")
        .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
        .group_by("user_id", "pixel_color")
        .agg(pl.count().alias("count"))
        .group_by("pixel_color")
        .agg(pl.col("count").count().alias("unique_count"))
        .sort("unique_count", descending=True)
        .head(3)
    )

    has_session = (
        pl.scan_parquet("2022pyarrow.parquet")
        .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
        .select(pl.col("user_id"))
        .group_by("user_id")
        .count()
        .sort("count", descending=True)
        .filter(pl.col("count") > 1)
    )

    session_lengths = (
        pl.scan_parquet("2022pyarrow.parquet")
        .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
        .filter(pl.col("user_id").is_in(has_session.select(pl.col("user_id")).collect()))
        .select(pl.col("user_id"), pl.col("timestamp"))
        .sort(["user_id", "timestamp"])
        .group_by("user_id") # get a list of timestamps for each user
        .agg([
            pl.col("timestamp"),
            (pl.col("timestamp")
            .diff()
            .fill_null(pl.duration(minutes=0)) 
            .alias("time_diff"))
        ])
        .explode(["timestamp", "time_diff"])  
        .with_columns(
            pl.when(pl.col("time_diff") >= pl.duration(minutes=15))
            .then(pl.duration(microseconds=0))
            .otherwise(pl.col("time_diff"))
            .alias("time_diff"),
            (pl.col("time_diff") > pl.duration(minutes=15)).cum_sum().alias("session_id")
        )
        .group_by(["user_id", "session_id"])
        .agg([
            (pl.col("timestamp").max() - pl.col("timestamp").min()).dt.total_seconds().alias("session_length (s)")
        ])
        .select(pl.col("session_length (s)")).mean()
    )

    pixels_placed_50 = (
        pl.scan_parquet("2022pyarrow.parquet")
        .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
        .select(pl.col("user_id"))
        .group_by("user_id")
        .count()
        .quantile(0.5)
    )

    pixels_placed_75 = (
        pl.scan_parquet("2022pyarrow.parquet")
        .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
        .select(pl.col("user_id"))
        .group_by("user_id")
        .count()
        .quantile(0.75)
    )

    pixels_placed_90 = (
        pl.scan_parquet("2022pyarrow.parquet")
        .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
        .select(pl.col("user_id"))
        .group_by("user_id")
        .count()
        .quantile(0.90)
    )

    pixels_placed_99 = (
        pl.scan_parquet("2022pyarrow.parquet")
        .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
        .select(pl.col("user_id"))
        .group_by("user_id")
        .count()
        .quantile(0.99)
    )

    first_time = (
        pl.scan_parquet("2022pyarrow.parquet")
        .select(pl.col("user_id"), pl.col("timestamp"))
        .group_by("user_id")
        .agg(pl.col("timestamp").sort().first().alias("first_timestamp"))
        .filter((pl.col("first_timestamp") >= start) & (pl.col("first_timestamp") <= end))
        .select(pl.col("first_timestamp"))
        .count()
    )

    print("Top 3 colors: ", ranks.collect())
    print("Average session: ", session_lengths.collect())
    print("50th percentile: ", pixels_placed_50.collect())
    print("75th percentile: ", pixels_placed_75.collect())
    print("90th percentile: ", pixels_placed_90.collect())
    print("99th percentile: ", pixels_placed_99.collect())
    print("First time users: ", first_time.collect())

    end_time = time.perf_counter_ns()
    elapsed_time_ms = (end_time - start_time) / 1000000
    print("Time taken: ", elapsed_time_ms, "ms")
