import pyarrow.csv as pv
import pyarrow.parquet as pq
import pyarrow as pa
import polars as pl

csv_file = "2022_place_canvas_history.csv"
# csv_file = "test.csv"
parquet_file = "2022pyarrow.parquet"

DATESTRING_FORMAT = "%Y-%m-%d %H:%M:%S"
BLOCK_SIZE = 100_000_000

read_options = pv.ReadOptions(block_size=BLOCK_SIZE)
csv_reader = pv.open_csv(csv_file, read_options=read_options)

parquet_writer = None

user_ids = {}
count = 0

try:
    for record_batch in csv_reader:
        print(f"Processing batch with {record_batch.num_rows} rows...")

        df = pl.from_arrow(record_batch)

        df = df.with_columns(
            pl.col("timestamp")
            .str.replace(r" UTC$", "")  
            .str.strptime(
                pl.Datetime, 
                format="%Y-%m-%d %H:%M:%S%.f",
                strict=False
            )
            .alias("timestamp")
        )

        df = (
            df.filter(
                pl.col("coordinate").str.count_matches(",") == 1
            )
            .with_columns(
                pl.col("coordinate")
                .str.split_exact(",", 1)
                .struct.field("field_0")
                .cast(pl.Int64)
                .alias("x"),
                pl.col("coordinate")
                .str.split_exact(",", 1)
                .struct.field("field_1")
                .cast(pl.Int64)
                .alias("y"),
            )
            .drop("coordinate")
            )

        # df = df.drop("user_id")

        ids = df["user_id"].unique()

        for i in range(len(ids)):
            if ids[i] not in user_ids:
                user_ids[ids[i]] = count
                count += 1

        df = df.with_columns(
            pl.col("user_id")
            .replace_strict(user_ids)
            .alias("user_id")
        )

        table = df.to_arrow()

        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                parquet_file, 
                schema=table.schema, 
                compression="zstd"
            )
        parquet_writer.write_table(table)

finally:
    if parquet_writer:
        parquet_writer.close()

print(f"Successfully converted {csv_file} to {parquet_file}")

print(df)
