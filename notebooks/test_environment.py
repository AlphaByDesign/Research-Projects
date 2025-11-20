import torch, polars as pl, altair as alt
print("torch ✅", torch.__version__)
print("polars ✅", pl.__version__)
print("altair ✅", alt.__version__)
x = torch.rand(2, 3)
print("tensor:", x)



from binance.client import Client
import pandas as pd
from datetime import datetime

# Initialize the Binance client (public data only)
client = Client()

# Parameters
symbol = "BTCUSDT"          # Change to any pair you want
interval = "1h"             # e.g. "1m", "15m", "1h", "1d"
start_date = "2020-01-01"   # You can use e.g. "1 Jan, 2020"
end_date = datetime.now().strftime("%d %b, %Y %H:%M:%S")

# Download historical klines
klines = client.get_historical_klines(symbol, interval, start_date, end_date)
