from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from keras import Input
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sqlite3

app = Flask(__name__)

# Load stock data
def load_stock_data(stock_symbol,start,end):
    data = yf.download(stock_symbol, start=start, end=end)
    return data['Close'].values.reshape(-1, 1), data

# Train LSTM Model
def train_model(stock_data):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    
    # Prepare the data for LSTM
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model using Input layer explicitly
    model = Sequential()

    # Define Input layer explicitly
    model.add(Input(shape=(x_train.shape[1], 1)))  # Explicit Input layer

    # Add LSTM layers
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))
    
    # Add Dense layers
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    return model, scaler

# Predict future prices
def predict_future(stock_data, model, scaler, days):
    last_60_days = stock_data[-60:]
    predicted_prices = []
    print(days,'======')
    for _ in range(days):
        scaled_last_60_days = scaler.transform(last_60_days)
        X_test = []
        X_test.append(scaled_last_60_days)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Predict the price for the next day
        predicted_price = model.predict(X_test)
        predicted_price_unscaled = scaler.inverse_transform(predicted_price)
        
        # Save the predicted price
        predicted_prices.append(predicted_price_unscaled[0, 0])
        
        # Update last_60_days to include the predicted price for the next prediction
        new_row = np.array([[predicted_price_unscaled[0, 0]]])
        last_60_days = np.append(last_60_days, new_row)[-60:].reshape(-1, 1)

    return predicted_prices

def save_stock_data(stock_symbol, historical_date, price, predicted_date, predicted_price, db_name="stock_data.db"):
   
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT NOT NULL,
            historical_date TEXT,
            price REAL,
            predicted_date TEXT,
            predicted_price REAL
        )
    """)
    print(stock_symbol, historical_date, price, predicted_date, predicted_price,'-=-=-=-=-=-=-=-=--=-=')
    # Insert the data into the table
    cursor.execute("""
        INSERT INTO stock_data (stock_symbol, historical_date, price, predicted_date, predicted_price)
        VALUES (?, ?, ?, ?, ?)
    """, (stock_symbol, historical_date, float(price), predicted_date,  float(predicted_price)))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()
    print(f"Data for stock '{stock_symbol}' saved successfully to {db_name}.")

# Graph creation
def create_graph(stock_data, predictions, stock_symbol):
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=(25, 14))

    # Extract the start and end dates
    start_date = stock_data.index[0].strftime('%Y-%m-%d')
    end_date = stock_data.index[-1].strftime('%Y-%m-%d')
    print(f"Stock Symbol: {stock_symbol}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")

    # Plot historical prices
    plt.plot(stock_data.index, stock_data['Close'], label='Historical Price', color='blue')

    # Predict future dates (e.g., if you predict 30 days ahead)
    last_date = stock_data.index[-1]
    predicted_dates = pd.date_range(last_date, periods=len(predictions) + 1, freq='B')[1:]  # Business days

    # Plot predicted prices with corresponding dates
    plt.plot(predicted_dates, predictions, label='Predicted Price', color='orange', alpha=0.9)

    plt.title(f"{stock_symbol} Stock Price Prediction\n\n(from {start_date}  to {end_date})", fontsize=18)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    # plt.grid(True)

    # # Display the graph
    # plt.show()

    # Save graph to a PNG image in memory
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    # fucntion to save data in databased
    save_stock_data(stock_symbol, start_date, predictions[0], end_date, predictions[-1])

    predicted_price = predictions[-1]
    print('Predicted Price : ',predicted_price)
    price = predictions[0]
    print(predicted_price,'=======++++======',price)
    percent_change = (predicted_price / price) * 100
    total_points_capture = predictions[-1] - predictions[0]
    total_profit_percent = round(percent_change, 2)
    print(total_points_capture, "...............",total_profit_percent)
    return graph_url,total_points_capture,total_profit_percent


@app.route('/')
def home():
    return render_template('index.html')

def test():
    return render_template('test.html')



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Use get_json to handle JSON input
    stock_symbol = data.get('stock')  # Get the stock symbol from the JSON data
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    days = int(data.get('days'))
    print('start Date : ',start_date,)
    print('End Date : ',end_date)
    print('Days : ',days)
    if not stock_symbol:
        return jsonify({'error': 'No stock symbol provided!'}), 400


    # Load stock data, train model, and predict
    stock_data, original_data = load_stock_data(stock_symbol,start_date,end_date)
    model, scaler = train_model(stock_data)

    # Predict the next 30 days
    predicted_prices = predict_future(stock_data, model, scaler, days=days)

    # Create a graph
    graph_url,total_points_capture,total_profit_percent = create_graph(original_data, predicted_prices, stock_symbol)

    return jsonify({'predicted_price': float(predicted_prices[-1]), 'graph_url': graph_url,'total_profit_percent':float(total_profit_percent),'total_points_capture':float(total_points_capture)})
    # return jsonify({'predicted_price': float(predicted_prices[0, 0]), 'graph_url': graph_url})



if __name__ == '__main__':
    app.run(debug=True)
