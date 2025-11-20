#%% md
# ### Creating a Time Series Model on BTCUSD on a 4H Time Horizon
# 
# * An Autoregression model (AR) with multiple lags was explored, including that of correlated coins.
# * Research found no significant alpha increase and thus kept features to its own timeseries.
# * Last feature to have been added was BTCUSD log volume, which increased the Sharpe by 0.4.
# 
# * We will be using a linear model inside Pytorch for the weight and bias.
#%%
from src import research
#%%

import polars as pl
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

research.set_seed(42)



from binance.client import Client
import pandas as pd
from datetime import datetime
import time

client = Client()

symbol = ["BTCUSDT"]
interval = "4h"
start_date = "2020-11-11"
end_date = datetime.now().strftime("%d %b, %Y %H:%M:%S")

def get_data(symbol):
    cols = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    df = pd.DataFrame(klines, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float, errors="ignore")
    df.set_index("timestamp", inplace=True)
    return df

# Loop through symbols and store in a dict
data = {}
for sym in symbols:
    print(f"Downloading {sym}...")
    data[sym] = get_data(sym)
    time.sleep(0.5)


# Combine both BTC and XRP close + volume into one DataFrame
prices = pd.concat(
    [
        data[sym][["close", "volume"]]
        .rename(columns={
            "close": f"{sym}_close",
            "volume": f"{sym}_volume"
        })
        for sym in symbols
    ],
    axis=1
)

print(prices.head())

#%% md
# ### Inspect data
#%%
prices.describe(include="all")
prices.isna().mean()
#%%
ts = prices
ts
#%% md
# ### Feature Engineering
# * Step 1:
# ### Create target and lagged features using log returns
#%%
# Creating log return within ts dataframe.
forcast_horizon = 1
#ts = ts.sort_index()
ts['close_log_return']= np.log(ts['BTCUSDT_close']/ts['BTCUSDT_close'].shift(forcast_horizon))
ts['log_volume'] = np.log(ts['BTCUSDT_volume']/ts['BTCUSDT_volume'].shift(forcast_horizon))
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

ts['log_volume_lag_1'] = ts['log_volume'].shift(forcast_horizon * 1)
ts['log_volume_lag_2'] = ts['log_volume'].shift(forcast_horizon * 2)
ts['log_volume_lag_3'] = ts['log_volume'].shift(forcast_horizon * 3)


#%%
# better practice would be ts = ts.dropna()
ts.dropna(inplace=True)
ts
#%% md
# ### Visualize the log returns distribution for outliers.
#%%
# Plot distribution

ts['close_log_return'].hist(bins=50, figsize=(10,5))
plt.title('Close log return')
plt.xlabel('Close log return')
plt.ylabel('Number of trades')

plt.show()

#%% md
# ### Constructing the Model
#%%
# we will use a linear model from torch.
# reason for linear model is the simplicity and interpretation
class LinearModel(nn.Module):
    def __init__(self, input_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return self.linear(x)
#%% md
# ### Splitting by time
# * Creating an (AR) model.
# * We are aiming to predict one return by its own lags.
# * splitting your data by scratch ensures no data leakage
#%%
 features = ['close_log_return_lag_1','close_log_return_lag_2','close_log_return_lag_3','log_volume_lag_1','log_volume_lag_2']
 target = 'close_log_return'
 test_size = 0.25 #
#%%
len(ts)
#%%
int(len(ts)* test_size)
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

# Replace old torch tensor
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
#%% md
# ### Training the model
# * I experimented with different loss functions and found the L1Loss to have the best performance.
#%%
# specify hyperparameters which can be tweaked to improve model performance

no_epochs = 1000 * 5
lr = 0.0005

# Create Model
model = LinearModel(len(features))

#Loss Function L1Loss/MSE / L1Loss has been the strongest performer through testing
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
# ### Model Trained: Looking at the weight, which is negative, the model picked up a mean reversion signal.
# ### Model Performance
# * Create trade results from testing data.
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
#trade_results
#%%
 # plot equity curve in log space
# Convert the Series to a NumPy array for plotting
y = trade_results['equity_curve'].to_numpy()

plt.figure(figsize=(10,5))
plt.plot(y)
plt.title("Equity Curve")
plt.xlabel("Index")
plt.ylabel("Equity")
plt.show()


#%% md
# ### Trade Performance Checking
#%%
# performance checking
trade_results = trade_results.with_columns(
    (pl.col('equity_curve')-pl.col('equity_curve').cum_max()).alias('drawdown_log')
)
trade_results
#%%
# Max DD log

max_drawdown_log = trade_results['drawdown_log'].min()
print('Max Drawdown Log:',max_drawdown_log)
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
print('Win Rate:',win_rate)
#%%
# Expected Value
avg_win = trade_results.filter(pl.col('is_won')==True)['trade_log_return'].mean()
avg_loss = trade_results.filter(pl.col('is_won')==False)['trade_log_return'].mean()
ev= win_rate * avg_win + (1-win_rate) * avg_loss
print('Expected Value:',ev)
#%%
# total log return
total_log_return = trade_results['trade_log_return'].sum()
total_log_return
#%%
compound_return = np.exp(total_log_return)
compound_return
#%%
1000*compound_return
print("Compound Return:", compound_return)
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
print('Std Deviation:',std)
#%%
# Sharpe
mean_ret = trade_results['trade_log_return'].mean()

# 4-hour bars = 6 periods per day * 365 days
annual_factor = np.sqrt(365 * 6)

sharpe = (mean_ret / std) * annual_factor
print("Sharpe Ratio:", sharpe)


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
trade_results

#%%
for x in trade_results['signal']:
    print(x)
#%% md
# 
# ### Tweaks:
# * I have adjusted the lag to 1 and put the time_interval to 4h. This has increased the sharpe and net equity.
#%%
y = trade_results['equity_curve_net'].to_numpy()

plt.figure(figsize=(10,5))
plt.plot(y)
plt.title("Equity Curve Net")
plt.xlabel("Index")
plt.ylabel("Equity")
plt.show()
#%%
torch.save(model.state_dict(), 'model_weights.pth')
#%%

#%% md
# ### Strategy Development
#%%
