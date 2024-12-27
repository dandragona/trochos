import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import random
import numpy as np
from tensorflow import keras
import tensorflow as tf
from xgboost import XGBClassifier, DMatrix
import xgboost as xgb
from collections import Counter
from sklearn.preprocessing import MinMaxScaler



class Stonks:
    def __init__(self, params):
        self.params = params
        self.model = None
    
    def load(self):
        rawDF = pd.read_parquet(f'ml/historical_data{"" if self.params['prod'] else "_single"}.parquet')
        print("Loaded:\n", rawDF.head())
        self.df = pre_process(rawDF,self.params['window_size'])

    def train_nn(self):        
        # Target variable
        print("Training on columns: ", self.df.columns)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        y = self.df['return_class'].cat.codes.to_numpy()

        # Select feature columns: exclude target and future_close
        feature_cols = [c for c in self.df.columns if c not in ['Symbol', 'date', 'Date', 'future_close', '7d_future_return', 'return_class']]
        X = self.df[feature_cols].values

        print("figuring out shape:", X.shape)

        #------------------------------------------------------------
        # Train/Validation/Test Split (time-based)
        #------------------------------------------------------------
        n = len(self.df)
        train_size = int(n * 0.85)
        val_size = int(n * 0.125)
        test_size = n - train_size - val_size
        print("(train, val, test):", (train_size, val_size, test_size))

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

        #------------------------------------------------------------
        # Build LSTM Model
        #------------------------------------------------------------
        num_features = X.shape[1]

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(num_features,)))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(4, activation='softmax'))  # 6 classes

        model.compile(
            loss='sparse_categorical_crossentropy',  # use 'categorical_crossentropy' if y is one-hot
            optimizer='adam',
            metrics=['sparse_categorical_accuracy']
        )

        model.summary()

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_val shape:", X_val.shape)
        print("y_val shape:", y_val.shape)
        
        class_counts = self.df['return_class'].value_counts()
        total = len(self.df)
        class_weight = {}
        for c, count in class_counts.items():
            class_weight[c] = total / (len(class_counts) * count)


        # Train the model
        history = model.fit(
            X_train, y_train,    # training data and labels
            validation_data=(X_val, y_val),  # validation set to monitor for overfitting
            epochs=5,            # adjust based on performance
            batch_size=128,         # can tune this
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            class_weight=class_weight,
            verbose=1
        )

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print("Test loss:", test_loss)
        print("Test accuracy:", test_acc)
        # print("Test precision:", test_precision)
        # print("Test accuracy:", test_recall)
        y_pred = model.predict(X_train)
        pairs = zip(y_pred, y_train)
        random_subset = random.sample(pairs, 50)
        # Print the sampled items
        for (pred, actual) in random_subset:
            print("Predicted: ", pred, np.argmax(pred),  "- Actual:", actual)

    def train_xgb(self):
        # Shuffle data so there is no date correlation.
        if self.params['shuffle']:
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        # Map the classes to integers
        class_map = {
            'large_negative': 0,
            'small_negative': 1,
            'small_positive': 2,
            'large_positive': 3
        }
        self.df['class_int'] = self.df['return_class'].map(class_map)


        # Scale the features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        feature_cols = [c for c in self.df.columns if c not in ['return_class', 'class_int', '5d_future_return', 'future_close']]
        self.df[feature_cols] = scaler.fit_transform(self.df[feature_cols])


        X = self.df[feature_cols]
        y = self.df['class_int']


        # Split into train and validation sets
        n = len(self.df)
        train_size = int(n * 0.85)
        val_size = int(n * 0.125)
        test_size = n - train_size - val_size
        print("(train, val, test):", (train_size, val_size, test_size))

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

        # Convert to DMatrix
        dtrain = DMatrix(X_train, label=y_train)
        dval = DMatrix(X_val, label=y_val)
        dtest = DMatrix(X_test, label=y_test)

        params = {
            'objective': 'multi:softmax',
            'num_class': 4,
            'learning_rate': self.params['learning_rate'],
            'max_depth': self.params['depth'],
            'eval_metric': 'mlogloss',
            'seed': 42
        }

        # Specify evaluation sets
        evals = [(dtrain, 'train'), (dval, 'eval')]

        # Train using xgb.train()
        # Note: n_estimators corresponds to num_boost_round
        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.params['num_iter'],
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=True
        )

        # Predict on validation set
        y_pred = bst.predict(dtest)

        # If you need to map predictions back to class names:
        inv_class_map = {v: k for k, v in class_map.items()}
        predicted_labels = [inv_class_map[i] for i in y_pred]
        actual_labels = [inv_class_map[i] for i in y_test]
        pairs = zip(predicted_labels, actual_labels)
        # Print the sampled items
        distribution = Counter(pairs)
        DistributionSummary(distribution)

        
        # Evaluate accuracy
        acc = accuracy_score(y_test, y_pred)
        with open('model_outputs.txt', 'a') as file:
            file.write(f"Test Accuracy: {acc:.5f}| Params: {self.params} | Stopped @ {bst.best_iteration}\n")
        print("Test Accuracy:", acc, "with params: ", self.params, "stopped at: ", bst.best_iteration)

    def train(self):
        #------------------------------------------------------------
        # Prepare X and y for Modeling
        #------------------------------------------------------------
        print("Training on columns: ", self.df.columns)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        y = self.df['7d_future_return'].values
        feature_cols = [c for c in self.df.columns if c not in ['date', 'future_close', '7d_future_return', 'Symbol']]
        X = self.df[feature_cols].values

        n = len(self.df)
        train_size = int(n * 0.9)
        val_size = int(n * 0.075)
        test_size = n - train_size - val_size
        print(f"Training with (train_size, val_size, test_size): {(train_size, val_size, test_size)}")
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

        #------------------------------------------------------------
        # Create and Train a Model (Using LightGBM)
        #------------------------------------------------------------
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            'objective': 'classification',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'num_leaves': 125,
            'max_depth': -1,
            'verbose': -1
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=10000,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.callback.log_evaluation(period=500)],
        )

        #------------------------------------------------------------
        # Evaluation on Test Set
        #------------------------------------------------------------
        y_pred = model.predict(X_test)
        print("Comparing predictions with actual:")
        for i in range(len(y_pred)):
            if random.random() < (50.0 / test_size):
                print(f"Predicted: {100*y_pred[i]:.2f} - Actual: {100*y_test[i]:.2f}")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("Test RMSE:", rmse)

        #------------------------------------------------------------
        # Further steps - Feature Importance
        #------------------------------------------------------------
        importance = model.feature_importance()
        feature_importance = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
        print("Feature importance:", feature_importance[:20])
        self.model = model

    def predict(self, ticker):
        input, err = self.get_prediction_input_for_ticker(ticker)
        if err != None:
            print(f"Prediction failed to generate data for ticker, returning err: {err}")
            return None, err
        return self.run_model_on_input(input)


def pre_process(df, window_size=15):
    print(f"Preprocessing: df.shape: {df.shape} with columns: {df.columns}")
    df.index.name = 'Date'
    df = df.sort_values(by=['Symbol', 'Date'])

    future_horizon = 5

    def compute_future_return_per_symbol(x):
        x['future_close'] = x['close'].shift(-future_horizon)
        x['5d_future_return'] = (x['future_close'] - x['close']) / x['close']
        return x

    df = df.groupby('Symbol', group_keys=False).apply(compute_future_return_per_symbol)


    # Drop rows where target can't be computed
    df.dropna(subset=['5d_future_return'], inplace=True)

    # Define bins and labels
    bins = [-np.inf, -0.05, 0, 0.05, np.inf]
    labels = ['large_negative', 'small_negative', 'small_positive', 'large_positive']

    df['return_class'] = pd.cut(df['5d_future_return'], bins=bins, labels=labels)
    class_distribution = df['return_class'].value_counts()
    print("Class distribution:", class_distribution)

    # Feature engineering:
    def add_features_per_symbol(x):
        # Let each sample have access to past info in window size
        lagged_features = {}
        for lag in range(1, window_size+1):
            lagged_features[f'Close_lag_{lag}'] = x['close'].shift(lag)
            lagged_features[f'Open_lag_{lag}'] = x['open'].shift(lag)
            lagged_features[f'High_lag_{lag}'] = x['high'].shift(lag)
            lagged_features[f'Low_lag_{lag}'] = x['low'].shift(lag)
            x[f'Volume_lag_{lag}'] = x['volume'].shift(lag)

        # # Moving averages
        lagged_features['MA_5'] = x['close'].rolling(5).mean()
        lagged_features['MA_10'] = x['close'].rolling(10).mean()
        lagged_features['MA_15'] = x['close'].rolling(15).mean()
        # Assume df is already loaded with columns: 'Open', 'High', 'Low', 'Close', 'Volume'
        # and sorted by Date in ascending order.

        # 1. Simple Moving Average (SMA)
        # Example: 20-day SMA
        window_sma = 20
        lagged_features['SMA_20'] = x['close'].rolling(window=window_sma).mean()

        # 2. Relative Strength Index (RSI)
        # RSI calculation steps:
        # - Compute daily returns
        # - Separate positive and negative returns
        # - Compute average gain and loss
        # - RSI = 100 - (100 / (1 + avg_gain/avg_loss))
        window_rsi = 14
        delta = x['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        gain_roll = pd.Series(gain, index=x.index).rolling(window=window_rsi).mean()
        loss_roll = pd.Series(loss, index=x.index).rolling(window=window_rsi).mean()

        rs = gain_roll / (loss_roll + 1e-10)  # add small epsilon to avoid division by zero
        lagged_features['RSI_14'] = 100 - (100 / (1 + rs))

        # 3. Moving Average Convergence Divergence (MACD)
        # MACD is the difference between two EMAs (commonly 12 and 26) of the Close, and a Signal line is EMA of MACD.
        short_ema_span = 12
        long_ema_span = 26
        signal_span = 9

        ema_short = x['close'].ewm(span=short_ema_span, adjust=False).mean()
        ema_long = x['close'].ewm(span=long_ema_span, adjust=False).mean()
        lagged_features['MACD'] = ema_short - ema_long
        lagged_features['MACD_Signal'] = lagged_features['MACD'].ewm(span=signal_span, adjust=False).mean()
        lagged_features['MACD_Hist'] = lagged_features['MACD'] - lagged_features['MACD_Signal']

        # 4. Bollinger Bands
        # Typically use a 20-day SMA and 2 standard deviations for upper/lower bands
        window_bb = 20
        lagged_features['BB_Mid'] = x['close'].rolling(window=window_bb).mean()
        lagged_features['BB_Std'] = x['close'].rolling(window=window_bb).std(ddof=0)
        lagged_features['BB_Upper'] = lagged_features['BB_Mid'] + 2 * lagged_features['BB_Std']
        lagged_features['BB_Lower'] = lagged_features['BB_Mid'] - 2 * lagged_features['BB_Std']

        # 5. On-Balance Volume (OBV)
        # OBV uses volume and price changes:
        # If today's close > yesterday's close: OBV = OBV_prev + Volume
        # Else if today's close < yesterday's close: OBV = OBV_prev - Volume
        # Else OBV = OBV_prev
        lagged_features['OBV'] = 0
        lagged_features['OBV'] = np.where(x['close'] > x['close'].shift(1),
                            x['volume'],
                            np.where(x['close'] < x['close'].shift(1),
                                    -x['volume'], 0))
        lagged_features['OBV'] = lagged_features['OBV'].cumsum()
        lagged_features['EMA_9'] = x['close'].ewm(9).mean().shift()

        # Daily returns & lagged returns

        x['daily_return'] = x['close'].pct_change()
        for lag in range(1, window_size+1):
            lagged_features[f'return_lag_{lag}'] = x['daily_return'].shift(lag)
        x = pd.concat([x, pd.DataFrame(lagged_features, index=x.index)], axis=1)
        return x

    df = df.groupby('Symbol', group_keys=False).apply(add_features_per_symbol)

    # Drop rows that don't have enough history
    df.dropna(inplace=True)

    # Since we have multiple symbols, we assume a global chronological split.
    # Sort by date one more time if needed.
    # Important: Ensure 'Date' is the index. If not, set it and sort.
    df = df.sort_index()  # ensures data sorted by Date
    print("df columns:", df.columns)
    return df

def categorize_percentage_change(percentage_change):
    if isinstance(percentage_change, np.ndarray):
        labels = np.zeros_like(percentage_change, dtype=int)
        labels[100*percentage_change <= -5] = 0
        labels[(100*percentage_change > -5) & (100*percentage_change < 0)] = 1
        labels[(100*percentage_change > 0) & (100*percentage_change < 5)] = 2
        labels[100*percentage_change >= 5] = 3
        return labels
    else:
        if 100*percentage_change <= -5:
            return 0
        elif 100*percentage_change > -5 and (100*percentage_change < 0):
            return 1
        elif 100*percentage_change >= 0 and (100*percentage_change < 5):
            return 2
        else:
            return 3

def DistributionSummary(dist):
    abbr = {
    'small_positive': 'Small+',
    'large_positive': 'Large+',
    'small_negative': 'Small-',
    'large_negative': 'Large-'
    }

    # List of classes
    classes = ['small_positive', 'large_positive', 'small_negative', 'large_negative']

    # Create the dictionary for all 16 pairs
    class_map = {
        (c1, c2): f"(Predicted: {abbr[c1]}, Actual: {abbr[c2]})"  # four spaces in between
        for c1 in classes
        for c2 in classes
    }
    for key, count in reversed(sorted(dist.items(), key=lambda x: x[1])):
        print(f"{class_map[key]} -> {count}")



if __name__ == "__main__":
    params = {
        'prod': False,
        'learning_rate': 0.01,
        'window_size': 15,
        'depth': 9,
        'num_iter': 4,
        'shuffle': True,
    }
    stonks = Stonks(params)
    stonks.load()
    stonks.train_xgb()