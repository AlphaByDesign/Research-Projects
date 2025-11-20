#%% md
# ### Hypothesis for Soybean directional, using the oilseeds complex as a lead driver.
# * We will test the relationship of the crush components of soybeans and its short term impact on directionality
#%% md
# ### Building a Linear Regression model for Soybeans
# * We will use Pytorch for the model
# * The variables will be the crush skew and crush for the dependents
# * We will perform a time series analysis using auto regression.
# 
#%% md
# # Importing Libraries
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import yfinance as yf


#%%
# Data retrieval & Inspect

tickers = ['ZS=F', 'ZM=F', 'ZL=F']
start_date = '2024-11-01'
end_date = '2025-11-10'
freq = '1h'

# Download both Close and Volume
data = yf.download(tickers, start=start_date, end=end_date, interval=freq)[['Close', 'Volume']]

# Flatten MultiIndex columns
data.columns = [f"{field}_{ticker.replace('=F', '')}" for field, ticker in data.columns]

# Ensure datetime index
data.index = pd.to_datetime(data.index)

data.head(25)

#data.describe()
#data.isna().mean()
#data.value_counts()


#%%
# visualize the dataframe
data['Close_ZS'].plot(figsize=(10,8), grid=True)

#%% md
# ### Feature engineering
# * Create target and lagged feature for the analysis
#%%
# Create crush skew formula

prices = data.copy()
prices['crush'] = (data['Close_ZS'] * 10) -(data['Close_ZM'] * 11) -(data['Close_ZL'] * 9)
prices['crush_skew'] = (data['Close_ZS'] * 6) -(data['Close_ZM'] * 11) -(data['Close_ZL'] * 9)
#prices['crush'].plot(figsize=(10,8), grid=True)
prices
#%%
target = 'crush_close_log_return'
max_lags = 4
forecast_horizon = 1

# Create log prices

prices = prices.sort_index()
prices['zs_close_log_return'] = np.log(prices['Close_ZS']/prices['Close_ZS'].shift(forecast_horizon))
prices['crush_close_log_return'] = np.log(prices['crush']/prices['crush'].shift(forecast_horizon))
prices['crush_skew_close_log_return'] = np.log(prices['crush_skew']/prices['crush_skew'].shift(forecast_horizon))

#(prices[['Close_ZS','crush','crush_skew']] <= 0).any()

# Create lagged features.
#prices[f'{target}_lag_1'] = prices[target].shift(forecast_horizon * 1)
#prices[f'{target}_lag_2'] = prices[target].shift(forecast_horizon * 2)
#prices[f'{target}_lag_3'] = prices[target].shift(forecast_horizon * 3)
#prices[f'{target}_lag_4'] = prices[target].shift(forecast_horizon * 4)

prices['crush_close_lag_1'] = prices['crush_close_log_return'].shift(forecast_horizon * 1)
prices['crush_close_lag_2'] = prices['crush_close_log_return'].shift(forecast_horizon * 2)
prices['crush_close_lag_3'] = prices['crush_close_log_return'].shift(forecast_horizon * 3)
prices['crush_close_lag_4'] = prices['crush_close_log_return'].shift(forecast_horizon * 4)

prices['crushskew_close_lag_1'] = prices['crush_skew_close_log_return'].shift(forecast_horizon * 1)
prices['crushskew_close_lag_2'] = prices['crush_skew_close_log_return'].shift(forecast_horizon * 2)
prices['crushskew_close_lag_3'] = prices['crush_skew_close_log_return'].shift(forecast_horizon * 3)
prices['crushskew_close_lag_4'] = prices['crush_skew_close_log_return'].shift(forecast_horizon * 4)

prices.dropna(inplace=True)
#%%
# Plot distribution of close log returns for Soybeans
prices['zs_close_log_return'].hist(bins=50, figsize=(5,4), grid=True)
plt.title('ZS Log Returns')
plt.xlabel('Log Returns')
plt.ylabel('Frequency')
plt.show()
#%%
prices['crush_close_log_return'].hist(bins=50, figsize=(5,4), grid=True)
plt.title('Crush Log Returns')
plt.xlabel('Log Returns')
plt.ylabel('Frequency')
plt.show()
#%%
prices['crush_skew_close_log_return'].hist(bins=50, figsize=(5,4), grid=True)
plt.title('CrushSkew Log Returns')
plt.xlabel('Log Returns')
plt.ylabel('Frequency')
plt.show()
#%% md
# ### Building the Model
#%%
# we will use a linear model from Pytorch.

class LinearModel(nn.Module):
    def __init__(self, input_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return self.linear(x)
#%% md
# ### Splitting the model by time to ensure no data leakage
#%%
features = ['crushskew_close_lag_1','crushskew_close_lag_2']
target = ['crush_close_log_return']
test_size = 0.25
#%%
len(prices)
#%%
len(prices) * test_size
#%%
# to split the data by time, we will split it by the index.
# this will give us the train size below

split_idx = int(len(prices) * (1-test_size))
split_idx
#%%
# split time series into 2 parts

prices_train, prices_test = prices[:split_idx], prices[split_idx:]

prices_train.head()
#%%
prices_test.head()
#%%
# We need to convert the split into sensors. Splitting our input and output into separate variables.

x_train = torch.tensor(prices_train[features].to_numpy(), dtype=torch.float32)
x_test = torch.tensor(prices_test[features].to_numpy(), dtype=torch.float32)
y_train = torch.tensor(prices_train[target].to_numpy(), dtype=torch.float32)
y_test = torch.tensor(prices_test[target].to_numpy(), dtype=torch.float32)
#%%
x_train.shape
#%%
y_test.shape
#%%
y_train.shape
#%%
# We need to fit the scaler

from sklearn.preprocessing import StandardScaler

x_scaler = StandardScaler().fit(prices_train[features].values)
X_train_np = x_scaler.transform(prices_train[features].values)
X_test_np  = x_scaler.transform(prices_test[features].values)

# Replace old torch tensor
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
#%% md
# ### Training the model
# 
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
    loss = criterion(y_hat, y_train)

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
        test_loss = criterion(y_hat, y_test)
        print(f'\nTest loss: {test_loss.item():.6f}, Train loss: {train_loss:.6f}')

#%% md
# # Evaluating trading performance
# * Create trade results from test data
#%%
# Create base DataFrame
trade_results = pd.DataFrame({
    'y_hat': np.squeeze(y_hat),
    'y': np.squeeze(y_test),
})

# Add calculated columns
trade_results['is_won'] = np.sign(trade_results['y_hat']) == np.sign(trade_results['y'])
trade_results['signal'] = np.sign(trade_results['y_hat'])
trade_results['trade_log_return'] = trade_results['signal'] * trade_results['y']
trade_results['equity_curve'] = trade_results['trade_log_return'].cumsum()

trade_results.head()

#%% md
# ### Visualize the equity curve
#%%
y = trade_results['equity_curve']
plt.figure(figsize=(10,5))
plt.plot(y)
plt.title('Equity Curve')
plt.xlabel('Index')
plt.ylabel('Equity')
plt.show()
#%%

#%%
