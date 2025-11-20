#%% md
# ### Creating a Time Series Model on BTCUSD
# 
# * Linear Autoregression (AR) with multiple lags.
# * We will be using a linear model but using Pytorch for the weight and bias
#%%
from binance.client import Client
import pandas as pd
from datetime import datetime
import polars as pl
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import altair as alt
import matplotlib.pyplot as plt


# Initialize the Binance client
client = Client()

# Parameters
symbol = "BTCUSDT"
interval = "4h"  # e.g. "1m", "15m", "1h", "1d"
start_date = "2020-01-01"  # You can use e.g. "1 Jan, 2020"
end_date = datetime.now().strftime("%d %b, %Y %H:%M:%S")
# Max number of Auto-regressive lags
max_lags =4
# Forecast horizon in steps
forcast_horizon = 1

# Download historical klines
klines = client.get_historical_klines(symbol, interval, start_date, end_date)

# Convert to DataFrame
cols = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

df = pd.DataFrame(klines, columns=cols)

# Convert datatypes
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
numeric_cols = ["open", "high", "low", "close", "volume"]
df[numeric_cols] = df[numeric_cols].astype(float)

# Set index
df.set_index("timestamp", inplace=True)

# Drop unused columns
df = df[["open", "high", "low", "close", "volume"]]

print(df.head())
#%% md
# ### Inspect data
#%%
df.describe(include="all")
df['close'].value_counts()
df.isna().mean()
#%%
ts = df
ts
#%%
ts['close'].plot(figsize=(10,5),title=symbol)
# can experiment with Altair for dynamic charts later.
#%% md
# ### Feature Engineering
# 
#%% md
# ### Create target and lagged features using log returns
#%%
# Creating log return within ts dataframe.
ts = ts.sort_index()
ts['close_log_return']= np.log(ts['close']/ts['close'].shift(forcast_horizon))
ts
#%%
# Create lagged features
target = 'close_log_return'
max_lags = 4
forcast_horizon = 1

# create 4 lagged features

ts = ts.copy() # this is to avoid setting with copy warning

ts[f'{target}_lag_1'] = ts[target].shift(forcast_horizon * 1)
ts[f'{target}_lag_2'] = ts[target].shift(forcast_horizon * 2)
ts[f'{target}_lag_3'] = ts[target].shift(forcast_horizon * 3)
ts[f'{target}_lag_4'] = ts[target].shift(forcast_horizon * 4)


#%%
# better practice would be ts = ts.dropna()
ts.dropna(inplace=True)
ts
#%%
# Plot distribution

ts['close_log_return'].hist(bins=50, figsize=(10,5))
plt.title('Close log return')
plt.xlabel('Close log return')
plt.ylabel('Number of trades')

plt.show()

#%%
import seaborn as sns
sns.histplot(ts['close_log_return'], bins=50, kde=True)
plt.title(" Distribution of Log Returns")

#%% md
# ### Build Model
#%%
# we will use a linear model from torch.
# reason for linear model is the simplicity and interpretation
class LinearModel(nn.Module):
    def __init__(self, input_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return self.linear(x)
#%%

#%% md
# ### Complexity of the model
#%%
# y = w * x + b = linear model
input_features = 1
linear_model = LinearModel(input_features)

from torchinfo import summary
#summary(linear_model,input_size=(1, input_features))

print(linear_model)
print(summary(linear_model,input_size=(1, input_features)))

#%%

#%% md
# ### Split by time
# * Creating an AR1 model
# * We are aiming to predict one return by its own lag
# * splitting your data by scratch ensures no data leackage
#%%
 features = ['close_log_return_lag_1']
 target = 'close_log_return'
 test_size = 0.25 #
#%%
len(ts)
#%%
len(ts)* test_size
#%%
# to split the data by time, we will split it by the index.
# this will give us the train size below
split_idx = int(len(ts) *(1-test_size))
split_idx
#%%
# split time series into 2 parts

ts_train,ts_test = ts[:split_idx], ts[split_idx:]

ts_train.head()
#%%
ts_test.tail()
#%%
# converting into torch tensors
# splitting our input and output into separate variables
X_train = torch.tensor(ts_train[features].to_numpy(), dtype=torch.float32)
X_test = torch.tensor(ts_test[features].to_numpy(), dtype=torch.float32)
Y_train = torch.tensor(ts_train[target].to_numpy(), dtype=torch.float32)
Y_test = torch.tensor(ts_test[target].to_numpy(), dtype=torch.float32)


#%%
X_train.shape # row vector

#%%
Y_train.shape # one dimensional tensor
#%%
# We need to put it into a 2 dimensional

Y_train = Y_train.reshape(-1, 1)
Y_train.shape
#%%
Y_test = Y_test.reshape(-1, 1)
Y_test.shape
#%%
# Fit scaler on TRAIN only
from sklearn.preprocessing import StandardScaler

x_scaler = StandardScaler().fit(ts_train[features].values)
X_train_np = x_scaler.transform(ts_train[features].values)
X_test_np  = x_scaler.transform(ts_test[features].values)

# Replace your old torch tensor creation with:
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32)

#%% md
# ### Batch Gradient Descent
# * this trains all the data at once.
#%%
# specify hyperparameters which can be tweaked to improve model performance

no_epochs = 1000 * 5
lr = 0.0005

# Create Model
model = LinearModel(len(features))

#Loss Function L1Loss/MSE
criterion = nn.L1Loss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

print('\nTraining...')
for epoch in range(no_epochs):
    # forward pass
    y_hat = model(X_train)
    loss = criterion(y_hat, Y_train)

    # Backward pass
    optimizer.zero_grad() # 1. clear old gradients
    loss.backward()       # 2. compute new gradients
    optimizer.step()      # 3. update weights

    # check for improvements by logging
    train_loss = loss.item()

    # logging
    if (epoch+1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{no_epochs}], Loss: {train_loss:.6f}')

    print('\nLearned parameters:')

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}:\n {param.data.numpy()}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_hat = model(X_test)
        test_loss = criterion(y_hat, Y_test)
        print(f'\nTest loss: {test_loss.item():.6f}, Train loss: {train_loss:.6f}')



#%% md
# ### Looking at the weight, which is negative, the model picked up a mean reversion adoption.
#%% md
# ### Test Trading performance
# * Create trade results from test data
#%%
trade_results = pl.DataFrame({
    'y_hat': y_hat.squeeze(),
    'y': Y_test.squeeze(),
}).with_columns(
    (pl.col('y_hat').sign()==pl.col('y').sign()).alias('is_won'),
    pl.col('y_hat').sign().alias('signal'),
).with_columns(
    (pl.col('signal') * pl.col('y')).alias('trade_log_return')
    ).with_columns(
        pl.col('trade_log_return').cum_sum().alias('equity_curve')
    )
trade_results
#%%
# plot equity curve
# Convert the Series to a NumPy array for plotting
y = trade_results['equity_curve'].to_numpy()

plt.figure(figsize=(10,5))
plt.plot(y)
plt.title("Equity Curve")
plt.xlabel("Index")
plt.ylabel("Equity")
plt.show()


#%%
# performance checking
trade_results = trade_results.with_columns(
    (pl.col('equity_curve')-pl.col('equity_curve').cum_max()).alias('drawdown_log')
)
trade_results
#%%
# Max DD log

max_drawdown_log = trade_results['drawdown_log'].min()
max_drawdown_log
#%%
# Putting into simple returns
drawdown_pct = np.exp(max_drawdown_log)-1
drawdown_pct
#%%
equity_peak = 1000
equity_peak * drawdown_pct
#%%
# Win rate
win_rate = trade_results['is_won'].mean()
win_rate
#%%
# Expected Value
avg_win = trade_results.filter(pl.col('is_won')==True)['trade_log_return'].mean()
avg_loss = trade_results.filter(pl.col('is_won')==False)['trade_log_return'].mean()
ev= win_rate * avg_win + (1-win_rate) * avg_loss
ev
#%%
# total log return
total_log_return = trade_results['trade_log_return'].sum()
total_log_return
#%%
compound_return = np.exp(total_log_return)
compound_return
#%%
1000*compound_return
#%%
# Equity trough

equity_trough = trade_results['equity_curve'].min()
equity_trough
#%%
# Equity peak

equity_peak = trade_results['equity_curve'].max()
equity_peak
#%%
# std
std = trade_results['trade_log_return'].std()
std
#%%
# Sharpe
annualized_rate = np.sqrt(365*24)
sharpe = ev / std * annualized_rate
sharpe

#%%
# Highlighting itertools
import itertools
max_lags = 4
benchmarks = []
feature_pool = [f'{target}_lag_{i}' for i in range(1, max_lags + 1)]
combos = list(itertools.combinations(feature_pool, 1))

for features in combos:
    model = LinearModel(len(features))
    benchmarks.append()(ts,list(features),target, model, annualized_rate,no_epochs=200,loss=nn.L1Loss())

    benchmarks = pl.DataFrame(benchmarks)



#%%

#%% md
# ### Adding Transaction Fees
#%%
# 2 types of fees:
# 1: maker fee and taker fee

maker_fee = 0.0001
taker_fee = 0.0003

roundtrip_fee_log = np.log(1-2*maker_fee)

trade_results = trade_results.with_columns(pl.lit(roundtrip_fee_log).alias('tx_fee_log'))
trade_results = trade_results.with_columns((pl.col('trade_log_return')+pl.col('tx_fee_log')).alias('trade_log_return_net'))
trade_results = trade_results.with_columns(pl.col('trade_log_return_net').cum_sum().alias('equity_curve_net'))
trade_results

roundtrip_fee_log
#trade_results

#%% md
# ### Our Initial time_interval resulted in a negative equity after fees. I adjusted in the following ways
# ### tweaks
# * Increased time horizon from 1H to 4H and used the lag_1 feature.After scanning through all single features, lag_1 resulted in highest performance
# * We can add further features next
# * We can explore correlation next
# 
# ### I have adjusted the lag to 1 and put the time_interval to 4h. This has increased the sharpe and net equity.
#%%
y = trade_results['equity_curve_net'].to_numpy()

plt.figure(figsize=(10,5))
plt.plot(y)
plt.title("Equity Curve Net")
plt.xlabel("Index")
plt.ylabel("Equity")
plt.show()
#%%
trade_results['is_won'].mean()
#%% md
# ### Now to create my own research module to adjust models and their functions without changing the full model.
# ### This will assist in identifying strength of model between features.
#%%
# Save
#%% md
# 