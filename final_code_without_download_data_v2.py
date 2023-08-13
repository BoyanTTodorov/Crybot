import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from joblib import load, dump
import tkinter as tk

# Function to load model from disk
def load_model(file_path):
    return load(file_path)

# Function to save model to disk
def save_model(model, file_path):
    dump(model, file_path)

# Function to save credentials to a JSON file
def save_credentials(api_key, api_secret):
    credentials = {'api_key': api_key, 'api_secret': api_secret}
    with open('credentials.json', 'w') as file:
        json.dump(credentials, file)

# Function to load credentials from a JSON file
def load_credentials():
    try:
        with open('credentials.json', 'r') as file:
            credentials = json.load(file)
            return credentials['api_key'], credentials['api_secret']
    except FileNotFoundError:
        return None, None
# Function to plot real-time chart with trades
def plot_real_time_chart_with_trades(data, predictions, trades):
    plt.figure(figsize=(15, 8))
    plt.plot(data['close'], label='Close Price', color='blue')
    for index, prediction in enumerate(predictions):
        color = 'orange' if prediction == 1 else 'purple'
        plt.scatter(index, data['close'].iloc[index], color=color, label='Prediction' if index == 0 else "")
    for trade in trades:
        index, action = trade
        color = 'green' if action == 'buy' else 'red'
        plt.scatter(index, data['close'].iloc[index], color=color, label='Trade' if index == trades[0][0] else "")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Real-time Bot Predictions and Trades')
    plt.legend(loc='upper left')
    plt.show(block=False)
    plt.pause(0.1)
    plt.clf()

# Trading bot class
class TradingBot:
    def __init__(self, symbol):
        self.symbol = symbol
        self.model = None
        self.is_trading = False

    def train_model(self, file_path):
        # Read data and prepare features and labels
        data = pd.read_csv(file_path)
        data['Label'] = (data['close'].shift(-1) > data['close']).astype(int)
        data['MA10'] = data['close'].rolling(window=10).mean()
        X = data[['open', 'high', 'low', 'close', 'MA10']].iloc[:-1].dropna()

        y = data['Label'].iloc[:-1].loc[X.index]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])

        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Set the trained model
        self.model = grid_search.best_estimator_

        # Evaluate the model on the testing set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")

        # Save the trained model to disk
        save_model(self.model, 'model.joblib')
        
    def start_trading(self):
        print("Starting trading...") # You can implement your trading logic here
        self.is_trading = True

    def stop_trading(self):
        print("Stopping trading...") # You can implement logic to stop trading here
        self.is_trading = False
# Trading bot GUI class
class TradingBotGUI(tk.Frame):
    def __init__(self, master=None, bot=None):
        super().__init__(master)
        self.master = master
        self.bot = bot
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # API key entry
        self.api_key_label = tk.Label(self, text='API Key:')
        self.api_key_label.pack()
        self.api_key_entry = tk.Entry(self)
        self.api_key_entry.pack()

        # API secret entry
        self.api_secret_label = tk.Label(self, text='API Secret:')
        self.api_secret_label.pack()
        self.api_secret_entry = tk.Entry(self, show='*')
        self.api_secret_entry.pack()

        # Train bot button
        self.train_bot_button = tk.Button(self, text='Train Bot', command=self.on_train_bot)
        self.train_bot_button.pack(pady=5)

        # Start/Stop trading button
        self.start_trading_button = tk.Button(self, text='Start Trading', command=self.on_start_trading)
        self.start_trading_button.pack(pady=5)

        # Quit button
        self.quit_button = tk.Button(self, text='QUIT', command=self.master.destroy)
        self.quit_button.pack(side='bottom')

    def on_train_bot(self):
        # Train the bot using pre-downloaded data
        file_path = 'Data Training\snall_XBTUSD_data.csv' # Replace with the actual path to the pre-downloaded data file
        self.bot.train_model(file_path)
        self.train_bot_button.config(text='Bot Trained')

    def on_start_trading(self):
        # Start or stop trading based on bot status
        if self.bot.is_trading:
            self.bot.stop_trading()
            self.start_trading_button.config(text='Start Trading')
        else:
            self.bot.start_trading()
            self.start_trading_button.config(text='Stop Trading')

# Create the trading bot instance
bot = TradingBot(symbol='XBTUSD')

# Create the GUI application
root = tk.Tk()
app = TradingBotGUI(master=root, bot=bot)
app.mainloop()
