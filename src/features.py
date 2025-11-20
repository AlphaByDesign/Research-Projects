import numpy as np
import polars as pl
from binance.client import Client
from datetime import datetime


# Define loading of data from Binance

client = Client()

def get_live_historical_data(
        symbol: str = "BTCUSDT",
        interval: str = "4h",
        start_date:str = "2020-11-11") -> pl.DataFrame:

    # using datetime.now to get the latest available data

        end_date = datetime.now().strftime("%d %b, %Y %H:%M:%S")
        klines = client.get_historical_klines(symbol,interval,start_date,end_date)

    # Defining the columns
        cols = ["date","open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
                "num_trades", "taker_buy_base", "taker_buy_quote", "ignore"]

    # Create Polars Dataframe
        df = pl.DataFrame(klines, schema=cols)

        df = df.with_columns([pl.col("date").cast(pl.Datetime(time_unit="ms")),
             pl.col(["open", "high", "low", "close", "volume"]).cast(pl.Float64)])

    # Select columns for analysis

        df = df.select([
        pl.col("date"),
        pl.col("open"),
        pl.col("high"),
        pl.col("low"),
        pl.col("close")])

        return df.sort("date")








