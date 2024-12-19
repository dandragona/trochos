import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np
from tensorflow import keras
import random


def load_df():
    # Load the text file (CSV format) into a DataFrame
    df = pd.read_csv('fake_data.txt', parse_dates=['Date'], index_col='Date')

    # Make sure the data is sorted by date (just in case)
    df.sort_index(inplace=True)

    print(df)
    return df

def run_lstm_model():
    #------------------------------------------------------------
    # Assume 'df' is prepared as before with columns including:
    # '7d_future_return' as the target, and lag features:
    # Close_lag_1, ..., Close_lag_15, Volume_lag_1, ..., Volume_lag_15, MA_5, MA_10, etc.
    #------------------------------------------------------------
    df = load_df()
    df = df.sort_index()

    # Define window_size (already known)
    window_size = 15

    # Target variable
    y = df['7d_future_return'].values

    # Select feature columns: exclude target and future_close
    feature_cols = [c for c in df.columns if c not in ['future_close', '7d_future_return']]
    X = df[feature_cols].values

    #------------------------------------------------------------
    # Reshaping X for LSTM
    #------------------------------------------------------------
    # Suppose you have created lag features consistently. For example, if your features are:
    # [Close_lag_1, Volume_lag_1, ..., Close_lag_15, Volume_lag_15, MA_5, MA_10, MA_15, ...]
    # You need to reshape them into (samples, timesteps, features_per_timestep).

    # Let's say you have total_features = X.shape[1].
    print("figuring out shape:", total_features.shape)
    total_features = X.shape[1]

    # For a time series model, you'd typically have something like:
    # features_per_timestep = total_features / window_size if all features are lag-based.
    # But if you have additional non-lag features, you'll need a consistent structure.
    # For example, if you have 2 main features (Close, Volume) lagged 15 times, that's 30 features,
    # plus 3 moving averages (MA_5, MA_10, MA_15) gives 33 features total.
    # One way is to arrange lags so that each timestep has the same feature set (e.g., Close, Volume, daily_return),
    # and moving averages could be either assigned to the last timestep or repeated.
    #
    # For simplicity, let's assume all features are lag-based and evenly divisible by window_size.
    # If not, you might need to reorganize your feature engineering step so each timestep has identical features.
    #
    # Example: If total_features = 30 and window_size = 15, features_per_timestep = 30/15 = 2.

    if total_features % window_size != 0:
        raise ValueError("Number of features not divisible by window_size. Restructure your features.")

    features_per_timestep = total_features // window_size
    X = X.reshape(X.shape[0], window_size, features_per_timestep)

    #------------------------------------------------------------
    # Train/Validation/Test Split (time-based)
    #------------------------------------------------------------
    n = len(df)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    test_size = n - train_size - val_size
    print("(train, val, test):", (train_size, val_size, test_size))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    #------------------------------------------------------------
    # Build LSTM Model
    #------------------------------------------------------------
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, input_shape=(window_size, features_per_timestep)))  
    model.add(keras.layers.Dense(1))  # Predict a single value: 7d future return

    model.compile(loss='mse', optimizer='adam')

    # Add callbacks such as early stopping
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #------------------------------------------------------------
    # Train the Model
    #------------------------------------------------------------
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    #------------------------------------------------------------
    # Evaluate on Test Set
    #------------------------------------------------------------
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Test RMSE:", rmse)
    print("Comparing predictions with actual:")
    for i in range(len(y_pred)):
        if random.random() < (50.0 / test_size):
            print(f"Predicted: {100*y_pred[i]:.2f} - Actual: {100*y_test[i]:.2f}")

def run_model():
    #------------------------------------------------------------
    # Step 1: Load and Prepare Data
    #------------------------------------------------------------

    # Suppose you have a DataFrame 'df' with:
    # Columns: 'Open', 'High', 'Low', 'Close', 'Volume', indexed by Date.
    # Example:
    # df = pd.read_csv('stock_data.csv', parse_dates=['Date'], index_col='Date')

    # For illustration, let's assume 'df' is already loaded:
    # df = ... # your loading code here

    # We need at least 'Close' prices to generate targets.
    # Ensure df is sorted by date
    df = load_df()
    print("df starts at: ", len(df), df.shape)

    #------------------------------------------------------------
    # Step 2: Define Target (7-day forward percentage change)
    #------------------------------------------------------------

    future_horizon = 7
    df['future_close'] = df['Close'].shift(-future_horizon)
    df['7d_future_return'] = (df['future_close'] - df['Close']) / df['Close']

    # Drop rows where target can't be computed
    df.dropna(subset=['7d_future_return'], inplace=True)

    #------------------------------------------------------------
    # Step 3: Feature Engineering
    # Using last 15 days (3 weeks approx.) of Close prices and some moving averages as features
    #------------------------------------------------------------
    window_size = 15

    # Create lag features for Close, Volume
    for lag in range(1, window_size+1):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        df[f'High_lag_{lag}'] = df['High'].shift(lag)
        df[f'Low_lag_{lag}'] = df['Low'].shift(lag)


    # Simple technical indicators: moving averages
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_15'] = df['Close'].rolling(15).mean()

    # Daily returns for the last window_size days
    df['daily_return'] = df['Close'].pct_change()
    for lag in range(1, window_size+1):
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)

    # Drop rows that don't have enough history
    df.dropna(inplace=True)

    #------------------------------------------------------------
    # Step 4: Prepare X and y for Modeling
    #------------------------------------------------------------
    y = df['7d_future_return'].values
    # Select all columns that are not the target or future_close
    feature_cols = [c for c in df.columns if c not in ['future_close', '7d_future_return']]
    X = df[feature_cols].values

    #------------------------------------------------------------
    # Step 5: Train/Validation/Test Split
    #
    # For time-series, ensure chronological splits. For simplicity:
    # - Train: first 70% of the data
    # - Validation: next 15%
    # - Test: last 15%
    # Adjust as per your needs.
    #------------------------------------------------------------
    n = len(df)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    test_size = n - train_size - val_size
    print("n, train_size, val_size, test_size: ", (n, train_size, val_size, test_size))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    #------------------------------------------------------------
    # Step 6: Create and Train a Model (Using LightGBM)
    #------------------------------------------------------------
    # LightGBM dataset format
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Set parameters (example; tune these)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': -1,
        'verbose': -1
    }

    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
    )

    #------------------------------------------------------------
    # Step 7: Evaluation on Test Set
    #------------------------------------------------------------
    print(X_test.shape)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("y_pred: ", y_pred, "Test RMSE:", rmse)

    #------------------------------------------------------------
    # Step 8: Further steps
    #
    # - Check feature importances
    # - Try different models or hyperparameters
    # - Consider a rolling CV validation scheme for more robust results
    #
    # Example: feature importance
    importance = model.feature_importance()
    feature_importance = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print("Feature importance:", feature_importance[:20])



def main():
    run_lstm_model()

if __name__ == "__main__":
    main()