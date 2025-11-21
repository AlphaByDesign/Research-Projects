import numpy as np
import polars as pl
from binance.client import Client
from datetime import datetime
from typing import List


# Define loading of data from Binance

client = Client()

def get_historical_data(
        symbol: str = "BTCUSDT",
        interval: str = "4h",
        start_date:str = "2020-11-11") -> pl.DataFrame:

    # Arguments:
    # symbol: symbol in str format
    # interval: specify the time horizon
    # start_date: starting date
    # end_date: ending date


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
        pl.col("close"),
        pl.col("volume")])

        return df.sort("date")


# Feature Engineering

def create_time_series_transform(
        df: pl.DataFrame,
        price_col: str = "close",
        volume_col: str = "volume",
        forecast_horizon: int = 1) -> pl.DataFrame:

# Calculates log returns for price and log volume changes.

# Arguments:
       # df: The input Polars DataFrame.
       # price_col: The price column ('close').
       # volume_col: The volume column ('volume').
       # forecast_horizon: The shift period for the return calculation.

# Returns:
       # The DataFrame with 'close_log_return' and 'log_volume' columns added.

# Calculate the log returns: log price/price.shift by forecast horizon

        log_returns = ((pl.col(price_col) / pl.col(price_col).shift(forecast_horizon)).log().alias("close_log_return"))

# Calculate Log Volume

        log_volume = ((pl.col(volume_col) / pl.col(volume_col).shift(forecast_horizon)).log().alias("log_volume"))


        df = df.with_columns(log_returns, log_volume)

        return df

# Create dynamic lag function. The functions aim is to loop through any number of lags required.

def create_lag_feature(
        df: pl.DataFrame,
        features: List[str],
        max_lags: int,
        forecast_horizon: int = 1) -> pl.DataFrame:

# Arguments:
        # df: the input df
        # A list of feature column names (eg."close_log_return")
        # max_lags: the max number of lags to create (n)
        # forecast_horizon: the base shift period

# Define the range of lag periods (1 through max lags)

        lag_periods = range(1,max_lags+1)
        lag_expressions = []

# Loop over features and period to build all expressions.

        for feature in features:
            for lag in lag_periods:
                expression = pl.col(feature).shift(forecast_horizon * lag).alias(f'{feature}_lag{lag}')
                lag_expressions.append(expression)

# Apply all lag expressions
        df = df.with_columns(lag_expressions)

# Drop all NaN from lagged/log features

        df = df.drop_nulls()

        return df





